"""Translating Iceberg file locations into paths DuckDB will accept.

Iceberg stores absolute locations in its manifests, and the exact spelling that
PyIceberg is happy with is not always one DuckDB understands. On Windows the two
disagree outright:

    PyIceberg  `PyArrowFileIO.parse_location` rebuilds the path as `netloc + path`,
               so a warehouse must be given as `file://C:/warehouse` (two slashes).
               `file:///C:/warehouse` yields `/C:/warehouse`, which Windows rejects.

    DuckDB     accepts `C:/x`, `C:\\x`, and `file:///C:/x` -- but *not* `file://C:/x`,
               the one form PyIceberg needs.

So a translation layer is mandatory, not cosmetic. Verified against pyiceberg 0.11.1
and duckdb 1.5.5 on Windows 10; `tests/unit/test_paths.py` pins every case.

Note on escaping: we deliberately do **not** percent-decode. PyIceberg's own
`parse_location` uses `urlparse` without unquoting, so a location it can read is
byte-identical to what we hand DuckDB. Partition values containing characters that
Iceberg percent-encodes are therefore untested until the integration run.
"""

from __future__ import annotations

import re

__all__ = ["engine_path", "engine_paths", "is_object_store", "scheme_of"]

# `s3a://`/`s3n://` are Hadoop-era spellings that appear in tables written by Spark.
# DuckDB's httpfs only knows `s3://`; the bucket/key after the scheme is identical.
_S3_ALIASES = ("s3a://", "s3n://")

_FILE_SCHEME = "file://"

# A path like `/C:/warehouse/...`: a Windows drive letter behind a leading slash,
# which is what stripping `file://` off `file:///C:/...` leaves behind.
_SLASH_DRIVE = re.compile(r"^/([A-Za-z]:[\\/])")

# Schemes DuckDB reaches over the network, and which therefore need httpfs loaded.
_OBJECT_STORE_SCHEMES = frozenset(
    {"s3", "s3a", "s3n", "gs", "gcs", "az", "abfs", "abfss", "http", "https"}
)


def scheme_of(location: str) -> str:
    """Return the URI scheme of `location`, or `""` for a bare filesystem path.

    A Windows drive letter is not a scheme, even though it parses like one:
    `C:/data` is a path, not a `c://` URI.
    """
    match = re.match(r"^([A-Za-z][A-Za-z0-9+.\-]*)://", location)
    return match.group(1).lower() if match else ""


def is_object_store(location: str) -> bool:
    """True when reading `location` needs DuckDB's httpfs extension."""
    return scheme_of(location) in _OBJECT_STORE_SCHEMES


def engine_path(location: str) -> str:
    """Translate one Iceberg location into a path DuckDB can read.

    >>> engine_path("s3a://bucket/key.parquet")
    's3://bucket/key.parquet'
    >>> engine_path("file://C:/warehouse/t/data/0.parquet")
    'C:/warehouse/t/data/0.parquet'
    >>> engine_path("file:///var/warehouse/t/data/0.parquet")
    '/var/warehouse/t/data/0.parquet'
    """
    for alias in _S3_ALIASES:
        if location.startswith(alias):
            return "s3://" + location[len(alias) :]

    if location.startswith(_FILE_SCHEME):
        rest = location[len(_FILE_SCHEME) :]
        # `file:///C:/x` -> `/C:/x` -> `C:/x`; a POSIX `file:///var/x` -> `/var/x` is
        # already correct and must keep its leading slash.
        if _SLASH_DRIVE.match(rest):
            rest = rest[1:]
        return rest

    return location


def engine_paths(locations: list[str]) -> list[str]:
    """Translate a list of Iceberg locations, preserving order."""
    return [engine_path(location) for location in locations]

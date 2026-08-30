"""Iceberg location -> DuckDB path translation.

The Windows cases are the reason this module exists: PyIceberg and DuckDB accept
disjoint spellings of a local file URI, so the translation is load-bearing, not
cosmetic. See the module docstring in `icetl.paths`.
"""

from __future__ import annotations

import pytest

from icetl.paths import engine_path, engine_paths, is_object_store, scheme_of


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        # Windows: `file://C:/x` is the only form PyIceberg accepts and the one form
        # DuckDB rejects, so it must lose the scheme.
        ("file://C:/warehouse/t/data/0.parquet", "C:/warehouse/t/data/0.parquet"),
        ("file:///C:/warehouse/t/data/0.parquet", "C:/warehouse/t/data/0.parquet"),
        ("file://D:/wh/x.parquet", "D:/wh/x.parquet"),
        # POSIX: the leading slash is part of the path and must survive.
        ("file:///var/warehouse/t/0.parquet", "/var/warehouse/t/0.parquet"),
        # Hadoop-era S3 spellings DuckDB's httpfs does not know.
        ("s3a://bucket/key.parquet", "s3://bucket/key.parquet"),
        ("s3n://bucket/key.parquet", "s3://bucket/key.parquet"),
        # Already fine, left alone.
        ("s3://bucket/key.parquet", "s3://bucket/key.parquet"),
        ("gs://bucket/key.parquet", "gs://bucket/key.parquet"),
        ("C:/warehouse/x.parquet", "C:/warehouse/x.parquet"),
        ("C:\\warehouse\\x.parquet", "C:\\warehouse\\x.parquet"),
        ("/var/warehouse/x.parquet", "/var/warehouse/x.parquet"),
        ("relative/x.parquet", "relative/x.parquet"),
    ],
)
def test_engine_path(location: str, expected: str) -> None:
    assert engine_path(location) == expected


def test_engine_path_is_idempotent() -> None:
    """Translating twice must not corrupt a path, since call sites may overlap."""
    for location in ("file://C:/wh/x.parquet", "s3a://b/k", "/var/x"):
        once = engine_path(location)
        assert engine_path(once) == once


def test_engine_paths_preserves_order() -> None:
    assert engine_paths(["s3a://b/2", "s3a://b/1"]) == ["s3://b/2", "s3://b/1"]


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("s3://b/k", "s3"),
        ("S3://b/k", "s3"),
        ("file:///x", "file"),
        ("abfss://c@a.dfs.core.windows.net/x", "abfss"),
        # A drive letter parses like a scheme but is not one.
        ("C:/warehouse", ""),
        ("C:\\warehouse", ""),
        ("/var/x", ""),
    ],
)
def test_scheme_of(location: str, expected: str) -> None:
    assert scheme_of(location) == expected


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("s3://b/k", True),
        ("s3a://b/k", True),
        ("gs://b/k", True),
        ("https://host/k", True),
        ("file:///var/x", False),
        ("C:/warehouse/x", False),
        ("/var/x", False),
    ],
)
def test_is_object_store(location: str, expected: bool) -> None:
    """Drives whether httpfs gets loaded, so a false positive costs a network call."""
    assert is_object_store(location) is expected

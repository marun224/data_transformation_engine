#!/usr/bin/env python
"""Thin runner for the wide-table benchmark harness.

    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --rows 2000000 --markdown BENCHMARKS.md
    uv run python scripts/benchmark.py --table nyc.yellow_tripdata

The implementation lives in `icetl.diagnostics.benchmark` so it is importable and
testable; this file only exists so the documented command works. It also runs
without an editable install, by putting `src/` on the path when needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from icetl.diagnostics.benchmark import main
except ModuleNotFoundError:  # pragma: no cover - only when run outside the venv
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from icetl.diagnostics.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())

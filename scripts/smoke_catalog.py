#!/usr/bin/env python
"""Thin runner for the connectivity smoke test.

    uv run python scripts/smoke_catalog.py
    uv run python scripts/smoke_catalog.py --namespace nyc --table yellow_tripdata -v

The implementation lives in `icetl.diagnostics.smoke` so it is importable and
testable; this file only exists so the documented command works. It also runs
without an editable install, by putting `src/` on the path when needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from icetl.diagnostics.smoke import main
except ModuleNotFoundError:  # pragma: no cover - only when run outside the venv
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from icetl.diagnostics.smoke import main

if __name__ == "__main__":
    raise SystemExit(main())

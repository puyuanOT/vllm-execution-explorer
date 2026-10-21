"""Controller, fixture lessons, and the live-engine session."""

from __future__ import annotations

import sys
from pathlib import Path

_instrumentation = Path(__file__).resolve().parents[1] / "instrumentation"
if _instrumentation.is_dir():
    _path = str(_instrumentation)
    if _path not in sys.path:
        sys.path.insert(0, _path)

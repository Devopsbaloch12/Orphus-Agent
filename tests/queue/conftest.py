"""Test bootstrap for ``tests/queue``.

Two temporary shims live here, both of which should disappear once the tree is
fully assembled:

1. ``src/`` is prepended to ``sys.path``. The project is not yet installed into
   the dev environment (``pip install -e .`` needs a ``README.md`` that does not
   exist yet), and ``pyproject.toml`` sets no ``pythonpath``.
2. ``orphus.observability.logging`` is stubbed onto the standard library logger
   when it is absent, because that module is being written in parallel.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:  # pragma: no cover - taken once the real module lands
    import orphus.observability.logging  # noqa: F401
except ImportError:  # pragma: no cover
    import orphus.observability as _observability

    _stub = types.ModuleType("orphus.observability.logging")
    _stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
    sys.modules["orphus.observability.logging"] = _stub
    _observability.logging = _stub  # type: ignore[attr-defined]

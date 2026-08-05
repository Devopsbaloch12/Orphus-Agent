"""Root pytest configuration.

Puts ``src/`` on ``sys.path`` so the suite runs against a working tree without
requiring ``pip install -e .`` first. That matters here because the test suite
is deliberately runnable on a machine with no CUDA, no model weights, and no
network -- the fewer setup steps between clone and green tests, the better.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

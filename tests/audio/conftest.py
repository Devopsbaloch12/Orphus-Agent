"""Make ``src/`` importable without an editable install.

Harmless once ``pip install -e .`` has been run: the path is only prepended when
it is not already resolvable.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

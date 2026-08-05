"""Bootstrap shim for the shared logger factory.

``orphus.observability.logging`` is owned by another workstream and may not be
importable yet. Every module in the model-adapter layer logs through this
function, which forwards to the real factory the moment it exists and falls back
to the stdlib root logger factory until then. The call signature and the return
type are identical, so deleting this module and importing
``orphus.observability.logging.get_logger`` directly is a mechanical change.

The lookup is done with :func:`importlib.import_module` rather than a top-level
``import`` so that neither ``mypy`` nor ``ruff`` has to resolve a package that
does not exist yet, and so that a partially-installed tree cannot break the
audio hot path at import time.
"""

from __future__ import annotations

import importlib
import logging
from typing import cast

__all__ = ["get_logger"]

_OBSERVABILITY_MODULE = "orphus.observability.logging"


def get_logger(name: str) -> logging.Logger:
    """Return the platform logger for ``name``.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A stdlib :class:`logging.Logger`. It accepts an ``extra=`` mapping on
        every call, whether it came from the observability package or the
        fallback.
    """
    try:
        module = importlib.import_module(_OBSERVABILITY_MODULE)
    except ImportError:
        return logging.getLogger(name)
    factory = module.get_logger
    return cast("logging.Logger", factory(name))

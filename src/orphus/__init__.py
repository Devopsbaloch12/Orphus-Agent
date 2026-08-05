"""Orphus: a production-grade local voice AI platform.

Streaming pipeline: microphone -> voice activity detection -> speech
recognition -> conversation manager -> LLM -> speech synthesis -> playback.

Nothing heavyweight is imported here. Importing ``orphus`` must stay cheap and
must never require CUDA, model weights, or a network, because the API layer,
the CLI, and the test suite all import it on machines that have none of those.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]

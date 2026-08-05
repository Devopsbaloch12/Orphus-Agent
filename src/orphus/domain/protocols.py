"""Structural interfaces for every swappable component.

These are ``typing.Protocol`` definitions, not abstract base classes. The
difference matters:

* Adapters (Silero, Nemotron, Orpheus, Grok) never import or inherit from
  platform code, so there is no cycle between ``domain`` and the adapter
  packages and no risk of a heavyweight GPU import leaking into the API layer.
* Test fakes satisfy the interface by shape alone. The entire suite therefore
  runs with no CUDA device, no model weights, and no network.
* ``mypy --strict`` still verifies conformance at every call site.

Every streaming method is an async generator so that backpressure propagates
naturally: a slow consumer stops pulling, and the producer stops producing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from orphus.domain.types import (
    AudioChunk,
    LLMUsage,
    Message,
    SessionId,
    TextDelta,
    TranscriptEvent,
    VadEvent,
)

__all__ = [
    "ASRSession",
    "HealthCheck",
    "HealthReport",
    "HealthStatus",
    "LLMProvider",
    "StreamingASR",
    "StreamingTTS",
    "TTSSession",
    "VadSession",
    "VoiceActivityDetector",
]

# ---------------------------------------------------------------------------
# Voice activity detection
# ---------------------------------------------------------------------------


@runtime_checkable
class VadSession(Protocol):
    """Per-session VAD state.

    Silero VAD carries recurrent hidden state across frames, so two concurrent
    conversations must never share one session object. The owning
    ``VoiceActivityDetector`` is shared; the session is not.
    """

    def process(self, chunk: AudioChunk) -> VadEvent | None:
        """Feed one frame and return a boundary event, if the frame crossed one.

        Args:
            chunk: Exactly ``VAD_FRAME_SAMPLES`` samples at ``VAD_SAMPLE_RATE``.

        Returns:
            A ``VadEvent`` on a speech-start or speech-end transition, otherwise
            ``None``. Steady-state frames deliberately produce no event to keep
            the hot path allocation-free.
        """
        ...

    def reset(self) -> None:
        """Clear recurrent state, e.g. after a barge-in or a stream restart."""
        ...


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """A loaded, shared VAD model that mints isolated per-session states."""

    def open_session(self, session_id: SessionId) -> VadSession:
        """Create an isolated VAD state for one conversation."""
        ...

    async def aclose(self) -> None:
        """Release the underlying model and its runtime."""
        ...


# ---------------------------------------------------------------------------
# Speech recognition
# ---------------------------------------------------------------------------


@runtime_checkable
class ASRSession(Protocol):
    """Per-session streaming ASR state.

    Cache-aware streaming models (Nemotron 3.5 / FastConformer-RNNT) keep an
    attention cache per stream. That cache is the session; the weights are not.
    """

    async def push(self, chunk: AudioChunk) -> None:
        """Submit audio for transcription. Returns as soon as it is enqueued."""
        ...

    def events(self) -> AsyncIterator[TranscriptEvent]:
        """Yield partial and final hypotheses as decoding progresses."""
        ...

    async def flush(self) -> TranscriptEvent | None:
        """Force the decoder to commit. Called on VAD speech-end.

        Returns:
            The final transcript for the utterance, or ``None`` if no speech was
            recognised.
        """
        ...

    async def aclose(self) -> None:
        """Release the attention cache and any queued work."""
        ...


@runtime_checkable
class StreamingASR(Protocol):
    """A loaded, GPU-resident ASR model serving many concurrent sessions."""

    def open_session(self, session_id: SessionId, *, language: str | None = None) -> ASRSession:
        """Create an isolated decoding stream.

        Args:
            session_id: Owning conversation.
            language: BCP-47 hint. ``None`` lets the multilingual model decide.
        """
        ...

    async def aclose(self) -> None:
        """Unload the model and free its VRAM."""
        ...


# ---------------------------------------------------------------------------
# Speech synthesis
# ---------------------------------------------------------------------------


@runtime_checkable
class TTSSession(Protocol):
    """One in-flight synthesis stream, cancellable mid-utterance."""

    def audio(self) -> AsyncIterator[AudioChunk]:
        """Yield synthesised PCM as it is decoded.

        The first chunk should arrive well before the last token is generated;
        that gap is what makes the conversation feel responsive.
        """
        ...

    async def cancel(self) -> None:
        """Abandon synthesis immediately.

        This is the barge-in path. It must be safe to call at any point,
        including before the first chunk and after the stream has finished.
        """
        ...


@runtime_checkable
class StreamingTTS(Protocol):
    """A loaded, GPU-resident TTS model serving many concurrent sessions."""

    def synthesize(
        self,
        text_stream: AsyncIterator[TextDelta],
        *,
        session_id: SessionId,
        voice: str | None = None,
    ) -> TTSSession:
        """Begin synthesising from a live stream of LLM text.

        Taking an ``AsyncIterator`` rather than a finished string is what allows
        synthesis to start on the first complete clause instead of waiting for
        the LLM to finish. Implementations are responsible for segmenting the
        incoming text at sensible prosodic boundaries.

        Args:
            text_stream: Assistant text as the LLM produces it.
            session_id: Owning conversation.
            voice: Model-specific voice name; ``None`` uses the configured default.
        """
        ...

    @property
    def available_voices(self) -> Sequence[str]:
        """Voice names this model accepts."""
        ...

    async def aclose(self) -> None:
        """Unload the model and free its VRAM."""
        ...


# ---------------------------------------------------------------------------
# Large language model
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """A chat-completion backend.

    Grok is the configured default, but no business logic may depend on that.
    Swapping providers must require only a config change and a new adapter.
    """

    @property
    def name(self) -> str:
        """Stable provider identifier, used in metrics and logs."""
        ...

    @property
    def model(self) -> str:
        """Model identifier currently in use."""
        ...

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        session_id: SessionId,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[TextDelta]:
        """Stream a completion token-by-token.

        Cancellation is expressed by closing the returned iterator (or
        cancelling the surrounding task); implementations must abort the
        underlying HTTP request rather than draining it, so that a barge-in
        stops burning provider tokens immediately.
        """
        ...

    async def last_usage(self, session_id: SessionId) -> LLMUsage | None:
        """Token usage for the most recent completion, if the provider reports it."""
        ...

    async def health(self) -> HealthReport:
        """Probe provider reachability without consuming meaningful quota."""
        ...

    async def aclose(self) -> None:
        """Close the underlying HTTP client and connection pool."""
        ...


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthStatus(Protocol):
    """Marker for the health enum, defined in ``orphus.observability.health``.

    Declared structurally here so ``domain`` stays free of imports from the
    observability package and the dependency graph remains acyclic.
    """

    value: str


@runtime_checkable
class HealthReport(Protocol):
    """Result of one component health probe."""

    @property
    def component(self) -> str:
        """Name of the probed component."""
        ...

    @property
    def healthy(self) -> bool:
        """Whether the component is serving correctly."""
        ...

    @property
    def detail(self) -> str | None:
        """Human-readable diagnosis when unhealthy."""
        ...


@runtime_checkable
class HealthCheck(Protocol):
    """Anything the health endpoint can probe."""

    async def check(self) -> HealthReport:
        """Run the probe. Must not raise; failures are reported, not thrown."""
        ...

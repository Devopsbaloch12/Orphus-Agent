"""Core domain types shared across every module.

This module is the root of the dependency graph: it imports nothing from
``orphus`` and everything else may import it. Keeping it dependency-free is what
prevents circular imports between the audio, asr, tts, and streaming packages.

Hot-path types (audio frames, transcript deltas, token deltas) are frozen
``dataclass``es with ``slots=True`` rather than Pydantic models. At 20 concurrent
sessions a 20ms frame cadence means ~1000 objects/second; Pydantic validation on
that path costs real CPU for no benefit, because the data never crosses a trust
boundary. Pydantic is used where it earns its keep: configuration and the API
edge, where input is untrusted.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, NewType, Self

import numpy as np
import numpy.typing as npt

__all__ = [
    "ASR_SAMPLE_RATE",
    "TTS_SAMPLE_RATE",
    "VAD_FRAME_SAMPLES",
    "VAD_SAMPLE_RATE",
    "AudioChunk",
    "AudioEncoding",
    "ConversationTurn",
    "LLMUsage",
    "Message",
    "MessageRole",
    "PcmArray",
    "SessionId",
    "SpeechSegment",
    "TextDelta",
    "TranscriptEvent",
    "TurnId",
    "VadEvent",
    "VadEventKind",
    "new_session_id",
    "new_turn_id",
]

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

SessionId = NewType("SessionId", str)
"""Opaque identifier for one voice conversation. Stable for the session's life."""

TurnId = NewType("TurnId", str)
"""Identifier for a single user-utterance -> assistant-reply exchange."""


def new_session_id() -> SessionId:
    """Mint a fresh session identifier."""
    return SessionId(f"sess_{uuid.uuid4().hex}")


def new_turn_id() -> TurnId:
    """Mint a fresh turn identifier."""
    return TurnId(f"turn_{uuid.uuid4().hex[:16]}")


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

PcmArray = npt.NDArray[np.float32]
"""Mono PCM samples normalised to [-1.0, 1.0]."""

VAD_SAMPLE_RATE: Final[int] = 16_000
"""Silero VAD operates only at 8kHz or 16kHz. We standardise on 16kHz."""

VAD_FRAME_SAMPLES: Final[int] = 512
"""Silero VAD v5 requires exactly 512 samples (32ms) per call at 16kHz."""

ASR_SAMPLE_RATE: Final[int] = 16_000
"""Nemotron 3.5 ASR expects 16kHz mono."""

TTS_SAMPLE_RATE: Final[int] = 24_000
"""Orpheus emits 24kHz mono via the SNAC decoder."""


class AudioEncoding(StrEnum):
    """Wire encodings accepted at the API edge."""

    PCM_S16LE = "pcm_s16le"
    PCM_F32LE = "pcm_f32le"


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """An immutable block of mono PCM audio.

    Args:
        samples: Mono float32 samples in [-1.0, 1.0].
        sample_rate: Samples per second.
        sequence: Monotonic index within the stream, used to detect gaps.
        captured_at: ``time.monotonic()`` at capture, for end-to-end latency
            measurement. Monotonic (not wall clock) because it is only ever
            used for durations and must not jump when NTP adjusts the clock.
    """

    samples: PcmArray
    sample_rate: int
    sequence: int = 0
    captured_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"expected mono 1-D audio, got shape {self.samples.shape}")
        if self.samples.dtype != np.float32:
            raise ValueError(f"expected float32 samples, got {self.samples.dtype}")

    @property
    def num_samples(self) -> int:
        """Number of PCM samples in this chunk."""
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        """Wall duration of this chunk in seconds."""
        return self.num_samples / self.sample_rate

    @classmethod
    def silence(cls, num_samples: int, sample_rate: int = ASR_SAMPLE_RATE) -> Self:
        """Build a chunk of digital silence. Used for padding and in tests."""
        return cls(samples=np.zeros(num_samples, dtype=np.float32), sample_rate=sample_rate)


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """A contiguous run of speech delimited by the VAD.

    Offsets are measured in seconds from the start of the session's audio stream,
    which makes them directly comparable across VAD, ASR, and transcript records.
    """

    start_s: float
    end_s: float
    audio: AudioChunk

    @property
    def duration_s(self) -> float:
        """Length of the speech segment in seconds."""
        return self.end_s - self.start_s


# ---------------------------------------------------------------------------
# Voice activity detection
# ---------------------------------------------------------------------------


class VadEventKind(StrEnum):
    """Transitions the VAD reports. Steady-state frames produce no event."""

    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"


@dataclass(frozen=True, slots=True)
class VadEvent:
    """A speech boundary detected in the inbound audio stream.

    ``SPEECH_STARTED`` is the barge-in trigger: the pipeline cancels any
    in-flight LLM generation and TTS playback the moment one arrives.
    """

    kind: VadEventKind
    timestamp_s: float
    probability: float


# ---------------------------------------------------------------------------
# Speech recognition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    """Incremental ASR output.

    Args:
        text: For a partial, the full hypothesis so far (not a delta) — this is
            what cache-aware RNNT decoding naturally produces and it keeps
            consumers from having to reassemble state. For a final, the
            committed transcript of the utterance.
        is_final: Whether the hypothesis is committed and will not change.
        confidence: Mean token confidence in [0, 1], or ``None`` when the
            decoder does not expose one.
        language: BCP-47 tag when the model reports detected language.
    """

    text: str
    is_final: bool
    timestamp_s: float
    confidence: float | None = None
    language: str | None = None


# ---------------------------------------------------------------------------
# Conversation and LLM
# ---------------------------------------------------------------------------


class MessageRole(StrEnum):
    """Chat roles, matching the OpenAI-compatible schema Grok exposes."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    """One entry in a conversation history."""

    role: MessageRole
    content: str

    def to_wire(self) -> dict[str, str]:
        """Render to the provider wire format."""
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A streamed fragment of assistant text from the LLM."""

    text: str
    is_final: bool = False


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Token accounting for one completion, when the provider reports it."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """A completed user/assistant exchange, durably recorded.

    Latencies are captured per stage so the benchmark suite and the Grafana
    dashboards read from the same source of truth rather than re-deriving it.
    """

    turn_id: TurnId
    session_id: SessionId
    user_text: str
    assistant_text: str
    started_at: float
    asr_latency_ms: float | None = None
    llm_first_token_ms: float | None = None
    tts_first_audio_ms: float | None = None
    end_to_end_ms: float | None = None
    usage: LLMUsage | None = None

    @property
    def messages(self) -> Sequence[Message]:
        """This turn rendered as chat messages for the next prompt."""
        return (
            Message(MessageRole.USER, self.user_text),
            Message(MessageRole.ASSISTANT, self.assistant_text),
        )

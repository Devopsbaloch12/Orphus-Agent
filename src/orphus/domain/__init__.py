"""Domain layer: core types and the structural interfaces between components.

This package is the root of the dependency graph. It imports nothing from the
rest of ``orphus``, and every other package may import it. That one rule is
what keeps the module graph acyclic.
"""

from __future__ import annotations

from orphus.domain.protocols import (
    ASRSession,
    HealthCheck,
    HealthReport,
    LLMProvider,
    StreamingASR,
    StreamingTTS,
    TTSSession,
    VadSession,
    VoiceActivityDetector,
)
from orphus.domain.types import (
    ASR_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    VAD_FRAME_SAMPLES,
    VAD_SAMPLE_RATE,
    AudioChunk,
    AudioEncoding,
    ConversationTurn,
    LLMUsage,
    Message,
    MessageRole,
    PcmArray,
    SessionId,
    SpeechSegment,
    TextDelta,
    TranscriptEvent,
    TurnId,
    VadEvent,
    VadEventKind,
    new_session_id,
    new_turn_id,
)

__all__ = [
    "ASR_SAMPLE_RATE",
    "TTS_SAMPLE_RATE",
    "VAD_FRAME_SAMPLES",
    "VAD_SAMPLE_RATE",
    "ASRSession",
    "AudioChunk",
    "AudioEncoding",
    "ConversationTurn",
    "HealthCheck",
    "HealthReport",
    "LLMProvider",
    "LLMUsage",
    "Message",
    "MessageRole",
    "PcmArray",
    "SessionId",
    "SpeechSegment",
    "StreamingASR",
    "StreamingTTS",
    "TTSSession",
    "TextDelta",
    "TranscriptEvent",
    "TurnId",
    "VadEvent",
    "VadEventKind",
    "VadSession",
    "VoiceActivityDetector",
    "new_session_id",
    "new_turn_id",
]

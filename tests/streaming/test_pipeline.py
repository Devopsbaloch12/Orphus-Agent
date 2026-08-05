from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from orphus.conversation.session import Session
from orphus.domain.types import (
    AudioChunk,
    LLMUsage,
    Message,
    SessionId,
    TextDelta,
    TranscriptEvent,
    VadEvent,
    VadEventKind,
)
from orphus.streaming import VoicePipeline


class FakeVad:
    def process(self, chunk: AudioChunk) -> VadEvent:
        return VadEvent(VadEventKind.SPEECH_ENDED, chunk.duration_s, 0.9)

    def reset(self) -> None:
        pass


class FakeAsr:
    async def push(self, chunk: AudioChunk) -> None:
        pass

    def events(self) -> AsyncIterator[TranscriptEvent]:
        raise NotImplementedError

    async def flush(self) -> TranscriptEvent:
        return TranscriptEvent("hello", True, 0.1)

    async def aclose(self) -> None:
        pass


class FakeLlm:
    name = "fake"
    model = "fake"

    async def _stream(self) -> AsyncIterator[TextDelta]:
        yield TextDelta("hi there")
        yield TextDelta("", True)

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        session_id: SessionId,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[TextDelta]:
        return self._stream()

    async def last_usage(self, session_id: SessionId) -> LLMUsage:
        return LLMUsage(2, 2)


class FakeTtsSession:
    def __init__(self, text: AsyncIterator[TextDelta], audio: AudioChunk) -> None:
        self.text = text
        self.chunk = audio
        self.cancelled = False

    async def _audio(self) -> AsyncIterator[AudioChunk]:
        async for _ in self.text:
            pass
        yield self.chunk

    def audio(self) -> AsyncIterator[AudioChunk]:
        return self._audio()

    async def cancel(self) -> None:
        self.cancelled = True


class FakeTts:
    available_voices = ("tara",)

    def __init__(self, audio: AudioChunk) -> None:
        self.audio_chunk = audio

    def synthesize(
        self,
        text_stream: AsyncIterator[TextDelta],
        *,
        session_id: SessionId,
        voice: str | None = None,
    ) -> FakeTtsSession:
        return FakeTtsSession(text_stream, self.audio_chunk)


async def test_complete_turn_reaches_playback_and_history() -> None:
    chunk = AudioChunk.silence(512)
    session = Session()
    pipeline = VoicePipeline(
        session=session, vad=FakeVad(), asr=FakeAsr(), llm=FakeLlm(), tts=FakeTts(chunk)
    )
    await pipeline.push_audio(chunk)
    await pipeline.wait_reply()
    assert await anext(pipeline.output()) is chunk
    assert session.turn_count == 1
    assert session.recent_turns[0].assistant_text == "hi there"
    await pipeline.aclose()


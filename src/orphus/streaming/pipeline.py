"""Per-session end-to-end streaming voice orchestration."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from orphus.conversation.session import Session
from orphus.domain.protocols import ASRSession, LLMProvider, StreamingTTS, TTSSession, VadSession
from orphus.domain.types import (
    AudioChunk,
    ConversationTurn,
    Message,
    MessageRole,
    TextDelta,
    VadEventKind,
    new_turn_id,
)
from orphus.queue import BoundedQueue, OverflowPolicy, QueueClosed

__all__ = ["VoicePipeline"]


class VoicePipeline:
    """Connect VAD, ASR, LLM and TTS for one isolated conversation."""

    def __init__(
        self,
        *,
        session: Session,
        vad: VadSession,
        asr: ASRSession,
        llm: LLMProvider,
        tts: StreamingTTS,
        voice: str | None = None,
        output_queue_size: int = 128,
        turn_sink: Callable[[ConversationTurn], Awaitable[None]] | None = None,
    ) -> None:
        self._session = session
        self._vad = vad
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._voice = voice
        self._output: BoundedQueue[AudioChunk] = BoundedQueue(
            output_queue_size,
            policy=OverflowPolicy.DROP_OLDEST,
            name="playback",
        )
        self._reply_task: asyncio.Task[None] | None = None
        self._tts_session: TTSSession | None = None
        self._turn_sink = turn_sink

    async def push_audio(self, chunk: AudioChunk) -> None:
        """Feed one model-ready audio frame and react to speech boundaries."""
        self._session.require_active()
        event = self._vad.process(chunk)
        if event is not None and event.kind is VadEventKind.SPEECH_STARTED:
            await self.interrupt()
        await self._asr.push(chunk)
        if event is not None and event.kind is VadEventKind.SPEECH_ENDED:
            transcript = await self._asr.flush()
            if transcript is not None and transcript.text.strip():
                await self.start_reply(transcript.text.strip())

    async def interrupt(self) -> None:
        """Stop provider generation and synthesis immediately on barge-in."""
        self._session.barge_in()
        if self._tts_session is not None:
            await self._tts_session.cancel()
        if self._reply_task is not None and not self._reply_task.done():
            self._reply_task.cancel()
            await asyncio.gather(self._reply_task, return_exceptions=True)
        self._reply_task = None
        self._tts_session = None

    async def start_reply(self, user_text: str) -> None:
        """Begin an LLM-to-TTS reply, replacing any existing one."""
        await self.interrupt()
        self._session.begin_turn()
        task = asyncio.create_task(self._run_reply(user_text))
        self._session.cancellation.bind(task)
        self._reply_task = task

    async def _run_reply(self, user_text: str) -> None:
        started = time.monotonic()
        messages = (*self._session.history.messages(), Message(MessageRole.USER, user_text))
        assistant_parts: list[str] = []

        async def text_stream() -> AsyncIterator[TextDelta]:
            async for delta in self._llm.stream_chat(
                messages, session_id=self._session.session_id
            ):
                if delta.text:
                    assistant_parts.append(delta.text)
                yield delta

        self._tts_session = self._tts.synthesize(
            text_stream(), session_id=self._session.session_id, voice=self._voice
        )
        async for chunk in self._tts_session.audio():
            self._session.cancellation.raise_if_cancelled()
            self._output.put(chunk)
        usage = await self._llm.last_usage(self._session.session_id)
        turn = ConversationTurn(
                turn_id=new_turn_id(),
                session_id=self._session.session_id,
                user_text=user_text,
                assistant_text="".join(assistant_parts),
                started_at=started,
                end_to_end_ms=(time.monotonic() - started) * 1000,
                usage=usage,
        )
        self._session.record_turn(turn)
        if self._turn_sink is not None:
            await self._turn_sink(turn)

    async def output(self) -> AsyncIterator[AudioChunk]:
        """Yield playback audio until the pipeline closes."""
        while True:
            try:
                yield await self._output.get()
            except QueueClosed:
                return

    async def wait_reply(self) -> None:
        if self._reply_task is not None:
            await self._reply_task

    async def aclose(self) -> None:
        await self.interrupt()
        await self._asr.aclose()
        self._vad.reset()
        self._output.close()

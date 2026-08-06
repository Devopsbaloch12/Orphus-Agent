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
from orphus.observability.logging import get_logger
from orphus.observability.metrics import PipelineStage, get_metrics
from orphus.queue import BoundedQueue, OverflowPolicy, PutOutcome, QueueClosed

__all__ = ["VoicePipeline"]

logger = get_logger(__name__)


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
        self._metrics = get_metrics()

    async def push_audio(self, chunk: AudioChunk) -> None:
        """Feed one model-ready audio frame and react to speech boundaries."""
        self._session.require_active()
        with self._metrics.time_stage(PipelineStage.VAD):
            event = self._vad.process(chunk)
        if event is not None:
            logger.info(f"pipeline.vad_event session={self._session.session_id} kind={event.kind}")
        if event is not None and event.kind is VadEventKind.SPEECH_STARTED:
            await self.interrupt()
        await self._asr.push(chunk)
        if event is not None and event.kind is VadEventKind.SPEECH_ENDED:
            # The moment the user actually stopped talking -- the anchor for
            # every downstream latency number, since it is what the user's
            # perceived "how long until it answered" is measured against.
            speech_ended_at = time.monotonic()
            with self._metrics.time_stage(PipelineStage.ASR):
                transcript = await self._asr.flush()
            asr_latency_ms = (time.monotonic() - speech_ended_at) * 1000
            logger.info(
                f"pipeline.asr_transcript session={self._session.session_id} "
                f"text={transcript.text if transcript else None!r} "
                f"latency_ms={asr_latency_ms:.1f}"
            )
            if transcript is not None and transcript.text.strip():
                await self.start_reply(transcript.text.strip(), speech_ended_at, asr_latency_ms)

    async def interrupt(self) -> None:
        """Stop provider generation and synthesis immediately on barge-in."""
        self._session.barge_in()
        if self._tts_session is not None:
            await self._tts_session.cancel()
        if self._reply_task is not None:
            # Check the result even if the task already finished on its own
            # (a naturally-failed task is "done" too) -- not just when we're
            # the one cancelling a still-running one. Gating the check on
            # `not done()` meant a reply that failed *before* the next
            # interrupt() call (e.g. the next turn's start_reply) would
            # never have its exception retrieved or logged at all.
            if not self._reply_task.done():
                self._reply_task.cancel()
            (result,) = await asyncio.gather(self._reply_task, return_exceptions=True)
            if isinstance(result, Exception):
                logger.exception(
                    f"pipeline.reply_task_failed session={self._session.session_id}",
                    exc_info=result,
                )
        self._reply_task = None
        self._tts_session = None

    async def start_reply(
        self, user_text: str, speech_ended_at: float, asr_latency_ms: float
    ) -> None:
        """Begin an LLM-to-TTS reply, replacing any existing one."""
        await self.interrupt()
        self._session.begin_turn()
        task = asyncio.create_task(self._run_reply(user_text, speech_ended_at, asr_latency_ms))
        self._session.cancellation.bind(task)
        self._reply_task = task

    async def _run_reply(
        self, user_text: str, speech_ended_at: float, asr_latency_ms: float
    ) -> None:
        dispatch_at = time.monotonic()
        messages = (*self._session.history.messages(), Message(MessageRole.USER, user_text))
        assistant_parts: list[str] = []
        llm_first_token_ms: float | None = None
        first_token_at: float | None = None

        async def text_stream() -> AsyncIterator[TextDelta]:
            nonlocal llm_first_token_ms, first_token_at
            async for delta in self._llm.stream_chat(
                messages, session_id=self._session.session_id
            ):
                if delta.text:
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                        llm_first_token_ms = (first_token_at - dispatch_at) * 1000
                        self._metrics.llm_first_token.labels(
                            provider=self._llm.name, model=self._llm.model
                        ).observe(llm_first_token_ms / 1000)
                    assistant_parts.append(delta.text)
                yield delta

        self._tts_session = self._tts.synthesize(
            text_stream(), session_id=self._session.session_id, voice=self._voice
        )
        tts_first_audio_ms: float | None = None
        end_to_end_ms: float | None = None
        async for chunk in self._tts_session.audio():
            self._session.cancellation.raise_if_cancelled()
            if end_to_end_ms is None:
                first_audio_at = time.monotonic()
                end_to_end_ms = (first_audio_at - speech_ended_at) * 1000
                self._metrics.end_to_end_latency.observe(end_to_end_ms / 1000)
                if first_token_at is not None:
                    tts_first_audio_ms = (first_audio_at - first_token_at) * 1000
                    self._metrics.tts_first_audio.observe(tts_first_audio_ms / 1000)
            outcome = self._output.put(chunk)
            self._metrics.queue_depth.labels(queue="playback").set(self._output.depth)
            self._metrics.queue_depth_observed.labels(queue="playback").observe(
                self._output.depth
            )
            if outcome is not PutOutcome.ACCEPTED:
                self._metrics.queue_dropped.labels(
                    queue="playback", policy=self._output.policy.value
                ).inc()
        usage = await self._llm.last_usage(self._session.session_id)
        turn = ConversationTurn(
                turn_id=new_turn_id(),
                session_id=self._session.session_id,
                user_text=user_text,
                assistant_text="".join(assistant_parts),
                started_at=speech_ended_at,
                asr_latency_ms=asr_latency_ms,
                llm_first_token_ms=llm_first_token_ms,
                tts_first_audio_ms=tts_first_audio_ms,
                end_to_end_ms=(
                    end_to_end_ms
                    if end_to_end_ms is not None
                    else (time.monotonic() - speech_ended_at) * 1000
                ),
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

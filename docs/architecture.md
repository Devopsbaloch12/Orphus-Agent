# Architecture

The FastAPI process owns one `ModelRuntime`. Startup loads Orpheus TTS,
Nemotron ASR, Silero VAD and the Grok connection pool exactly once. Uvicorn
must remain at one worker because another process would duplicate GPU weights.

Each REST-created conversation owns a `Session`, history, cancellation token,
VAD state, ASR buffer and `VoicePipeline`. Binary 16 kHz PCM16 WebSocket frames
are decoded, resampled and framed before VAD/ASR. Speech end commits ASR text;
Grok deltas feed clause-sized Orpheus synthesis; 24 kHz PCM16 is returned over
the same socket. Speech start cancels provider generation and synthesis.

Bounded queues shed stale media rather than accumulating latency. A bounded
priority worker pool rejects excess work instead of growing memory. Redis is
the hot-state boundary and PostgreSQL stores completed turns and stage latency.
The provider interface is structural, so Grok can be replaced without changing
conversation orchestration.


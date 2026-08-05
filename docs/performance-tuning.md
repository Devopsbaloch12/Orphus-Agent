# Performance tuning

Measure first-audio latency and queue depth under the actual voice distribution.
Start with 20 admitted sessions, ASR context `[56, 3]`, one API process and TTS
GPU utilization 0.45. Reduce TTS reservation if ASR allocation fragments; lower
ASR right context for latency, accepting accuracy loss. Never raise queue sizes
to conceal sustained overload.

The initial ASR adapter commits each VAD-delimited utterance through the
Transformers RNNT API. For maximum 20-session throughput, replace its inference
method with NeMo's cache-aware streaming pipeline and microbatch active streams;
the `StreamingASR` boundary and session ownership do not change.

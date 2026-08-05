# Configuration

Precedence is default YAML, environment YAML, local YAML, dotenv, process
environment, then runtime overrides. Use nested variables such as
`ORPHUS_TTS__GPU_MEMORY_UTILIZATION=0.40`; documented flat aliases including
`GROK_API_KEY`, `REDIS_URL`, `DATABASE_URL`, `HOST` and `PORT` are supported.

Production requires both `GROK_API_KEY` and `ORPHUS_API_KEYS`. Never place
secrets in committed YAML. Model paths default to `models/{vad,asr,tts}` and
may be overridden independently. Keep `server.api_workers=1` for in-process GPU
models.


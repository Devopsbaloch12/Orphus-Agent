# Operations runbook

## Startup and validation

Run `./deploy.sh`, then `./scripts/status.sh`. Readiness requires the API,
Redis, PostgreSQL, CUDA and all model loaders. A WebSocket close code 1013 means
the model runtime did not load; inspect `./scripts/logs.sh api --tail 200`.

## Common incidents

- CUDA OOM: stop admission, restart the API, lower TTS GPU memory utilization,
  and reduce concurrency. Do not add Uvicorn workers because each duplicates
  model weights.
- Grok failures: the circuit breaker fails requests quickly and resets after
  the configured interval. Check the key, base URL and outbound connectivity.
- Growing latency: inspect queue depth/drop metrics. Reduce admission or ASR
  right context before increasing queue capacity.
- Database unavailable: preserve live conversation service when possible and
  alert on failed turn persistence. Run `alembic current` before migrations.

## Safe restart

Use `./scripts/restart.sh`. It drains the process within the configured grace
period, cancels active generations, closes model sessions, and lets Supervisor
restart failed processes.

## Rollback

Keep the previous Git revision and virtual environment until health checks pass.
Checkout the previous revision, reinstall its locked dependencies, run only
backward-compatible migrations, and restart. Database downgrade is a separate,
explicit operation and should not be automatic.

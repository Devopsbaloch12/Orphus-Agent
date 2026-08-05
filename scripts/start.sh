#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/start.sh -- start the Orphus process group under supervisor.
#
# Purpose:
#   Bring up supervisord (if it is not already running) and start the managed
#   programs: orphus-api, plus orphus-redis / orphus-postgres when this host
#   manages its own datastores.
#
#   Safe to run repeatedly. Programs that are already RUNNING are left alone.
#
#   The environment matters here: supervisord's own process environment is what
#   every child inherits, and that is how flat variables (GROK_API_KEY,
#   REDIS_URL, ORPHUS_API_KEYS, ...) reach the application. This script exports
#   .env and .env.secrets before starting supervisord for exactly that reason.
#   Consequence: after editing either file you must restart supervisord itself,
#   not just the program -- use ./scripts/restart.sh --reload-env.
#
# Usage:
#   ./scripts/start.sh [service] [--wait] [--timeout <s>] [--help]
#
# Exit status:
#   0 on success, non-zero if supervisord or a requested service fails to start.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

# shellcheck source=_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
install_err_trap "start.sh"

SERVICE="all"
WAIT=0
TIMEOUT="${ORPHUS_HEALTH_TIMEOUT_S:-300}"

usage() {
    cat <<'EOF'
start.sh -- start the Orphus process group.

USAGE
    ./scripts/start.sh [service] [options]

ARGUMENTS
    service        orphus-api | orphus-redis | orphus-postgres | all
                   (default: all)

OPTIONS
    --wait         Block until GET /health returns 2xx.
    --timeout <s>  Budget for --wait (default 300; a cold vLLM load is slow).
    -h, --help     Show this message.

EXAMPLES
    ./scripts/start.sh
    ./scripts/start.sh --wait
    ./scripts/start.sh orphus-api --wait --timeout 600
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait)      WAIT=1 ;;
        --timeout)   TIMEOUT="${2:?--timeout needs a value}"; shift ;;
        -h|--help)   usage; exit 0 ;;
        -*)          log_error "unknown option: $1"; echo; usage; exit 2 ;;
        *)           SERVICE="$1" ;;
    esac
    shift
done

load_orphus_env

[[ -f "${SUPERVISOR_CONF}" ]] \
    || die "no supervisor configuration at ${SUPERVISOR_CONF}. Run ./deploy.sh first."

if supervisor_running; then
    log_ok "supervisord already running (pid $(sctl pid))"
else
    log_info "starting supervisord"
    ensure_dir "${LOG_DIR}" "${RUN_DIR}"
    "$(supervisord_bin)" -c "${SUPERVISOR_CONF}" \
        || die "supervisord failed to start. Inspect ${LOG_DIR}/supervisord.log"
    # supervisord daemonises immediately; give it a moment to bind the socket.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        supervisor_running && break
        sleep 1
    done
    supervisor_running || die "supervisord did not create ${SUPERVISOR_SOCK}. See ${LOG_DIR}/supervisord.log"
    log_ok "supervisord up"
fi

log_info "starting ${SERVICE}"
# supervisorctl exits non-zero for "already started", which is the idempotent
# case rather than an error. Inspect the output instead of the exit status.
out="$(sctl start "${SERVICE}" 2>&1 || true)"
printf '%s\n' "${out}" | sed 's/^/         /'

if printf '%s' "${out}" | grep -qE 'ERROR|FATAL|BACKOFF|abnormal'; then
    if printf '%s' "${out}" | grep -qi 'already started'; then
        log_ok "already running"
    else
        log_error "one or more services failed to start:"
        sctl status >&2 || true
        log_error "  ./scripts/logs.sh api --tail 100"
        exit 1
    fi
fi

sctl status || true

if [[ "${WAIT}" -eq 1 ]]; then
    log_info "waiting up to ${TIMEOUT}s for ${BASE_URL}/health"
    attempts=$(( TIMEOUT / 5 )); [[ "${attempts}" -ge 1 ]] || attempts=1
    if wait_for_http "${BASE_URL}/health" "${attempts}" 5; then
        log_ok "API healthy at ${BASE_URL}"
    else
        log_error "API did not become healthy. See ./scripts/logs.sh api --tail 100"
        exit 1
    fi
fi

log_ok "start complete"

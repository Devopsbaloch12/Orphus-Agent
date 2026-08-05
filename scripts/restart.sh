#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/restart.sh -- restart Orphus services.
#
# Purpose:
#   Two distinct restarts, because they are not interchangeable:
#
#     (default)        supervisorctl restart -- reloads code and weights, keeps
#                      supervisord and its inherited environment.
#     --reload-env     full stop, supervisord shutdown, start. Required after
#                      editing .env or .env.secrets, because the application
#                      reads flat variables from supervisord's *inherited*
#                      process environment. A plain restart re-executes the
#                      child with the OLD environment and silently ignores the
#                      change -- the single most common "my config edit did
#                      nothing" report.
#
#   A restart is not free: the API reloads a 3B TTS model into vLLM plus the
#   ASR checkpoint, which costs roughly 60-180s of unavailability. Live calls
#   are dropped. Prefer a maintenance window.
#
# Usage:
#   ./scripts/restart.sh [service] [--reload-env] [--wait] [--help]
#
# Exit status:
#   0 on success, non-zero if the service does not come back.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
install_err_trap "restart.sh"

SERVICE="all"
RELOAD_ENV=0
WAIT=0
TIMEOUT="${ORPHUS_HEALTH_TIMEOUT_S:-300}"

usage() {
    cat <<'EOF'
restart.sh -- restart Orphus services.

USAGE
    ./scripts/restart.sh [service] [options]

ARGUMENTS
    service         orphus-api | orphus-redis | orphus-postgres | all
                    (default: all)

OPTIONS
    --reload-env    Stop supervisord entirely and start it again so that
                    changes to .env / .env.secrets take effect. Use this after
                    ANY configuration edit.
    --wait          Block until GET /health returns 2xx.
    --timeout <s>   Budget for --wait (default 300).
    -h, --help      Show this message.

EXAMPLES
    ./scripts/restart.sh orphus-api --wait
    ./scripts/restart.sh --reload-env --wait
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reload-env) RELOAD_ENV=1 ;;
        --wait)       WAIT=1 ;;
        --timeout)    TIMEOUT="${2:?--timeout needs a value}"; shift ;;
        -h|--help)    usage; exit 0 ;;
        -*)           log_error "unknown option: $1"; echo; usage; exit 2 ;;
        *)            SERVICE="$1" ;;
    esac
    shift
done

load_orphus_env

start_args=()
[[ "${WAIT}" -eq 1 ]] && start_args+=(--wait --timeout "${TIMEOUT}")

if [[ "${RELOAD_ENV}" -eq 1 ]]; then
    log_step "Full restart with environment reload"
    log_info "supervisord will be shut down so children pick up the new .env"
    "${SCRIPT_DIR}/stop.sh" --shutdown
    "${SCRIPT_DIR}/start.sh" "${start_args[@]+"${start_args[@]}"}"
    log_ok "restarted with a fresh environment"
    exit 0
fi

log_step "Restarting ${SERVICE}"
if ! supervisor_running; then
    log_warn "supervisord is not running; starting instead of restarting"
    exec "${SCRIPT_DIR}/start.sh" "${SERVICE}" "${start_args[@]+"${start_args[@]}"}"
fi

log_warn "the API reloads all models on restart; expect 60-180s of downtime"
sctl restart "${SERVICE}" 2>&1 | sed 's/^/         /' || true
sctl status || true

if [[ "${WAIT}" -eq 1 ]]; then
    attempts=$(( TIMEOUT / 5 )); [[ "${attempts}" -ge 1 ]] || attempts=1
    log_info "waiting up to ${TIMEOUT}s for ${BASE_URL}/health"
    if wait_for_http "${BASE_URL}/health" "${attempts}" 5; then
        log_ok "API healthy at ${BASE_URL}"
    else
        log_error "API did not come back. ./scripts/logs.sh api --tail 100"
        exit 1
    fi
fi

log_ok "restart complete"

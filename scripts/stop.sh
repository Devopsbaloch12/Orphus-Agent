#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/stop.sh -- stop the Orphus process group.
#
# Purpose:
#   Gracefully stop managed services. The API is given server.shutdown_grace_s
#   (default 20s, supervisor waits 45s) to drain in-flight conversations before
#   it is killed, so callers are not cut off mid-sentence.
#
#   Safe to run when nothing is running.
#
# Usage:
#   ./scripts/stop.sh [service] [--shutdown] [--force] [--help]
#
# Exit status:
#   0 on success (including "already stopped"), non-zero if a process would not
#   die even after --force.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

# shellcheck source=_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
install_err_trap "stop.sh"

SERVICE="all"
SHUTDOWN=0
FORCE=0

usage() {
    cat <<'EOF'
stop.sh -- stop the Orphus process group.

USAGE
    ./scripts/stop.sh [service] [options]

ARGUMENTS
    service        orphus-api | orphus-redis | orphus-postgres | all
                   (default: all)

OPTIONS
    --shutdown     Also stop supervisord itself. Required before changing
                   .env, because children inherit supervisord's environment.
    --force        SIGKILL anything still alive after the graceful window,
                   including orphaned vLLM engine workers holding VRAM.
    -h, --help     Show this message.

EXAMPLES
    ./scripts/stop.sh
    ./scripts/stop.sh orphus-api
    ./scripts/stop.sh --shutdown --force
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shutdown) SHUTDOWN=1 ;;
        --force)    FORCE=1 ;;
        -h|--help)  usage; exit 0 ;;
        -*)         log_error "unknown option: $1"; echo; usage; exit 2 ;;
        *)          SERVICE="$1" ;;
    esac
    shift
done

load_orphus_env

force_kill_strays() {
    # stopasgroup/killasgroup in the program config should make this a no-op.
    # It is here for the case where supervisord itself died and left the engine
    # workers behind, which is the usual reason a restart fails with a CUDA OOM.
    local pids
    pids="$(pgrep -f 'orphus.api.app|vllm|VLLM::EngineCore' 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        log_warn "killing stray model processes: ${pids//$'\n'/ }"
        # shellcheck disable=SC2086
        kill -9 ${pids} 2>/dev/null || true
        sleep 2
    fi
}

if ! supervisor_running; then
    log_ok "supervisord is not running; nothing to stop"
    if [[ "${FORCE}" -eq 1 ]]; then force_kill_strays; fi
    exit 0
fi

log_info "stopping ${SERVICE} (graceful window: 45s)"
sctl stop "${SERVICE}" 2>&1 | sed 's/^/         /' || true
sctl status || true

if [[ "${SHUTDOWN}" -eq 1 ]]; then
    log_info "shutting down supervisord"
    sctl shutdown 2>&1 | sed 's/^/         /' || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        supervisor_running || break
        sleep 1
    done
    if supervisor_running; then
        log_warn "supervisord still up after 10s"
        if [[ "${FORCE}" -eq 1 ]]; then
            pkill -f "supervisord -c ${SUPERVISOR_CONF}" || true
        fi
    else
        log_ok "supervisord stopped"
    fi
fi

if [[ "${FORCE}" -eq 1 ]]; then
    force_kill_strays
    if gpu_present; then
        log_info "VRAM after stop:"
        gpu_memory_report | sed 's/^/         /'
        log_info "If used memory is still high, a stray process holds it: nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    fi
fi

log_ok "stop complete"

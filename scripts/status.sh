#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# scripts/status.sh -- one screen answering "is this box healthy?"
#
# Purpose:
#   Process state, GPU/VRAM, datastore reachability, model inventory, disk, and
#   the API's own view of itself. Read-only: it never starts, stops, or
#   modifies anything.
#
# Usage:
#   ./scripts/status.sh [--json] [--quiet] [--help]
#
# Exit status:
#   0 everything up
#   1 at least one component is down   (usable as a cron/monitor probe)
#   2 bad usage
# ---------------------------------------------------------------------------
set -Eeuo pipefail

# shellcheck source=_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
install_err_trap "status.sh"

JSON=0
QUIET=0

usage() {
    cat <<'EOF'
status.sh -- Orphus service, GPU, and dependency status.

USAGE
    ./scripts/status.sh [options]

OPTIONS
    --json      Machine-readable summary (for monitors and CI).
    --quiet     Suppress the human report; exit status only.
    -h, --help  Show this message.

EXIT STATUS
    0  all components up
    1  at least one component down
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)    JSON=1 ;;
        --quiet)   QUIET=1 ;;
        -h|--help) usage; exit 0 ;;
        *)         log_error "unknown option: $1"; echo; usage; exit 2 ;;
    esac
    shift
done

load_orphus_env

DOWN=0
declare -A STATE=()

record() { STATE["$1"]="$2"; [[ "$2" == "up" ]] || DOWN=1; }

say()     { [[ "${QUIET}" -eq 1 || "${JSON}" -eq 1 ]] || printf '%b\n' "$*"; }
section() { [[ "${QUIET}" -eq 1 || "${JSON}" -eq 1 ]] || log_step "$*"; }

# --- Supervisor -------------------------------------------------------------
section "Services"
if supervisor_running; then
    record supervisord up
    say "  supervisord   pid $(sctl pid)"
    if [[ "${QUIET}" -eq 0 && "${JSON}" -eq 0 ]]; then
        sctl status 2>/dev/null | sed 's/^/  /' || true
    fi
    for svc in "${SVC_API}" "${SVC_REDIS}" "${SVC_POSTGRES}"; do
        line="$(sctl status "${svc}" 2>/dev/null || true)"
        [[ -n "${line}" ]] || continue
        grep -qE '(^|[[:space:]])RUNNING' <<< "${line}" && record "${svc}" up || record "${svc}" down
    done
else
    record supervisord down
    say "  ${C_RED}supervisord is not running${C_RESET}  ->  ./scripts/start.sh"
fi

# --- API --------------------------------------------------------------------
section "API"
health_code="$(http_status "${BASE_URL}/health" 5)"
ready_code="$(http_status "${BASE_URL}/health/ready" 10)"
metrics_code="$(http_status "${BASE_URL}${ORPHUS_METRICS_PATH:-/metrics}" 5)"

say "  liveness      ${BASE_URL}/health          -> ${health_code}"
say "  readiness     ${BASE_URL}/health/ready    -> ${ready_code}"
say "  metrics       ${BASE_URL}${ORPHUS_METRICS_PATH:-/metrics}  -> ${metrics_code}"
[[ "${health_code}" =~ ^2 ]] && record api up || record api down
[[ "${ready_code}"  =~ ^2 ]] && record api_ready up || record api_ready down

if [[ "${health_code}" =~ ^2 && "${QUIET}" -eq 0 && "${JSON}" -eq 0 ]]; then
    body="$(http_get "${BASE_URL}/health" 5 2>/dev/null || true)"
    [[ -n "${body}" ]] && printf '%s\n' "${body}" | sed 's/^/  /'
fi

# --- GPU --------------------------------------------------------------------
section "GPU"
if gpu_present; then
    record gpu up
    if [[ "${QUIET}" -eq 0 && "${JSON}" -eq 0 ]]; then
        printf '  %-4s %-28s %10s %10s %6s\n' IDX NAME USED TOTAL UTIL
        gpu_memory_report | while IFS=',' read -r idx name used total util; do
            printf '  %-4s %-28s %10s %10s %6s\n' \
                "$(echo "${idx}" | xargs)" "$(echo "${name}" | xargs)" \
                "$(echo "${used}" | xargs)" "$(echo "${total}" | xargs)" \
                "$(echo "${util}" | xargs)"
        done
        # Per-process VRAM: this is how you find the stray engine worker that
        # is holding memory after a bad shutdown.
        procs="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true)"
        if [[ -n "${procs}" ]]; then
            say "  compute processes:"
            printf '%s\n' "${procs}" | sed 's/^/    /'
        else
            say "  compute processes: none (no model is loaded)"
        fi
    fi
else
    record gpu down
    say "  ${C_YELLOW}no NVIDIA GPU visible${C_RESET}"
fi

# --- Datastores -------------------------------------------------------------
section "Datastores"
r_host="$(redis_host)"; r_port="$(redis_port)"
p_host="$(pg_host)";    p_port="$(pg_port)"
if tcp_open "${r_host}" "${r_port}" 3; then record redis up; else record redis down; fi
if tcp_open "${p_host}" "${p_port}" 3; then record postgres up; else record postgres down; fi
say "  redis         ${r_host}:${r_port}    $(mark "$([[ ${STATE[redis]} == up ]] && echo 0 || echo 1)")"
say "  postgres      ${p_host}:${p_port}    $(mark "$([[ ${STATE[postgres]} == up ]] && echo 0 || echo 1)")"

# --- Models -----------------------------------------------------------------
section "Models"
for comp in vad asr tts snac tts-tokenizer; do
    dir="${MODEL_ROOT}/${comp}"
    if [[ -d "${dir}" ]] && [[ -n "$(ls -A "${dir}" 2>/dev/null)" ]]; then
        size="$(du -sh "${dir}" 2>/dev/null | cut -f1)"
        repo="$(head -1 "${dir}/.orphus-complete" 2>/dev/null || echo 'no sentinel')"
        say "  $(printf '%-14s' "${comp}") ${C_GREEN}present${C_RESET}  ${size:-?}  ${repo}"
        record "model_${comp}" up
    else
        say "  $(printf '%-14s' "${comp}") ${C_RED}missing${C_RESET}  ${dir}"
        record "model_${comp}" down
    fi
done

# --- Host -------------------------------------------------------------------
section "Host"
if [[ "${QUIET}" -eq 0 && "${JSON}" -eq 0 ]]; then
    df -h "${MODEL_ROOT}" "${LOG_DIR}" 2>/dev/null | sed 's/^/  /' || true
    say "  load          $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo n/a)"
    if [[ -r /proc/meminfo ]]; then
        say "  memory        $(awk '/MemAvailable/ {printf "%.1f GiB available", $2/1048576}' /proc/meminfo)"
    fi
fi

# --- Output -----------------------------------------------------------------
if [[ "${JSON}" -eq 1 ]]; then
    printf '{'
    first=1
    for k in "${!STATE[@]}"; do
        [[ "${first}" -eq 1 ]] || printf ','
        printf '"%s":"%s"' "${k}" "${STATE[$k]}"
        first=0
    done
    printf ',"overall":"%s"}\n' "$([[ "${DOWN}" -eq 0 ]] && echo up || echo degraded)"
elif [[ "${QUIET}" -eq 0 ]]; then
    echo
    if [[ "${DOWN}" -eq 0 ]]; then
        log_ok "all components up"
    else
        log_warn "one or more components are down; see docs/runbook.md"
    fi
fi

exit "${DOWN}"

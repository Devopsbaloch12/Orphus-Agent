# shellcheck shell=bash
# ---------------------------------------------------------------------------
# scripts/_common.sh -- shared helpers for every Orphus operational script.
#
# Purpose:
#   Path resolution, .env loading, colored logging, and supervisor plumbing,
#   in one place so the nine operator scripts stay short and behave identically.
#
# Usage:
#   Sourced, never executed:
#       source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
#   The sourcing script owns `set -Eeuo pipefail`; this file deliberately does
#   not set shell options on its caller's behalf.
#
# Every path and tunable below is read from the environment with a sane
# default. Nothing here hardcodes a secret, an absolute path, or a GPU id.
# ---------------------------------------------------------------------------

# --- Paths ------------------------------------------------------------------
# Resolve the repo root from this file's location, so scripts work regardless
# of the caller's working directory.
_common_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORPHUS_ROOT="${ORPHUS_ROOT:-$(cd "${_common_dir}/.." && pwd)}"
unset _common_dir

export ORPHUS_ROOT
VENV_DIR="${ORPHUS_VENV:-${ORPHUS_ROOT}/.venv}"
RUN_DIR="${ORPHUS_RUN_DIR:-${ORPHUS_ROOT}/run}"
LOG_DIR="${LOG_DIR:-${ORPHUS_ROOT}/logs}"
MODEL_ROOT="${MODEL_ROOT:-${ORPHUS_ROOT}/models}"
STATE_DIR="${RUN_DIR}/state"

SUPERVISOR_CONF="${ORPHUS_SUPERVISOR_CONF:-${RUN_DIR}/supervisord.conf}"
SUPERVISOR_SOCK="${RUN_DIR}/supervisor.sock"
SUPERVISOR_PID="${RUN_DIR}/supervisord.pid"
SUPERVISOR_INCLUDE_DIR="${RUN_DIR}/supervisor.d"

# Process group and the individual program names, as they appear in
# `supervisorctl status`.
SUPERVISOR_GROUP="orphus"
SVC_API="orphus-api"
SVC_REDIS="orphus-redis"
SVC_POSTGRES="orphus-postgres"

# --- Logging ----------------------------------------------------------------
# Colors only when attached to a terminal and NO_COLOR is unset, so piped and
# CI output stays free of escape sequences.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    C_RESET=$'\033[0m'
    C_RED=$'\033[0;31m'
    C_GREEN=$'\033[0;32m'
    C_YELLOW=$'\033[0;33m'
    C_BLUE=$'\033[0;34m'
    C_CYAN=$'\033[0;36m'
    C_BOLD=$'\033[1m'
    C_DIM=$'\033[2m'
else
    C_RESET='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE='' C_CYAN='' C_BOLD='' C_DIM=''
fi

_ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

log_step()  { printf '\n%s==>%s %s%s%s\n'  "${C_BLUE}"  "${C_RESET}" "${C_BOLD}" "$*" "${C_RESET}"; }
log_info()  { printf '%s[ info ]%s %s\n'   "${C_CYAN}"  "${C_RESET}" "$*"; }
log_ok()    { printf '%s[  ok  ]%s %s\n'   "${C_GREEN}" "${C_RESET}" "$*"; }
log_warn()  { printf '%s[ warn ]%s %s\n'   "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
log_error() { printf '%s[ fail ]%s %s\n'   "${C_RED}"   "${C_RESET}" "$*" >&2; }
log_debug() {
    [[ -n "${ORPHUS_DEBUG:-}" ]] || return 0
    printf '%s[debug ]%s %s %s\n' "${C_DIM}" "${C_RESET}" "$(_ts)" "$*" >&2
}

# Print an error and exit non-zero. Always give the operator the next action.
die() {
    log_error "$*"
    exit 1
}

# Install a uniform ERR trap. Call as: install_err_trap "<script name>"
install_err_trap() {
    local name="${1:-${BASH_SOURCE[1]##*/}}"
    # shellcheck disable=SC2064  # expand $name now, not at trap time
    trap "_orphus_on_err \"${name}\" \$LINENO \$? \"\$BASH_COMMAND\"" ERR
}

_orphus_on_err() {
    local name="$1" line="$2" code="$3" cmd="$4"
    log_error "${name}: failed at line ${line} (exit ${code})"
    log_error "  command: ${cmd}"
    exit "${code}"
}

# --- Environment ------------------------------------------------------------
# Parse a dotenv file without executing it. Only `KEY=value` lines are honored;
# surrounding single/double quotes are stripped; `export ` prefixes tolerated.
# Values are NOT expanded, so a `$` or a backtick in a secret is safe.
load_env_file() {
    local file="$1" line key value
    [[ -f "${file}" ]] || return 0
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%$'\r'}"                       # tolerate CRLF
        [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
        line="${line#"${line%%[![:space:]]*}"}"    # ltrim
        line="${line#export }"
        [[ "${line}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
        key="${line%%=*}"
        value="${line#*=}"
        # Strip one layer of matched quotes.
        if [[ "${value}" == \"*\" || "${value}" == \'*\' ]]; then
            value="${value:1:${#value}-2}"
        fi
        # Process environment wins over the file, matching settings.py
        # precedence (env > .env).
        if [[ -z "${!key+set}" ]]; then
            export "${key}=${value}"
        fi
    done < "${file}"
    log_debug "loaded env from ${file}"
}

# Load .env, then .env.secrets, then re-derive the paths they may have changed.
#
# Two files, on purpose. pydantic-settings parses `.env` itself with
# extra="forbid", and raises SettingsError on any ORPHUS_-prefixed key it
# cannot map to a top-level Settings field. `ORPHUS_API_KEYS` is exactly such a
# key -- it is a flat alias for security.api_keys, resolved by
# FlatEnvAliasSource from the *process environment*. So it must be exported,
# not written into .env. `.env.secrets` is that export-only file; the Python
# side never reads it. See docs/configuration.md.
load_orphus_env() {
    load_env_file "${ORPHUS_ROOT}/.env"
    load_env_file "${ORPHUS_ROOT}/.env.secrets"
    export ORPHUS_ENV="${ORPHUS_ENV:-production}"
    LOG_DIR="${LOG_DIR:-${ORPHUS_ROOT}/logs}"
    MODEL_ROOT="${MODEL_ROOT:-${ORPHUS_ROOT}/models}"
    HOST="${HOST:-0.0.0.0}"
    PORT="${PORT:-8000}"
    # Never probe 0.0.0.0; it is a bind address, not a destination.
    PROBE_HOST="${ORPHUS_PROBE_HOST:-127.0.0.1}"
    BASE_URL="${ORPHUS_BASE_URL:-http://${PROBE_HOST}:${PORT}}"
    export LOG_DIR MODEL_ROOT HOST PORT
}

# --- Small utilities --------------------------------------------------------
have() { command -v "$1" >/dev/null 2>&1; }

require_cmd() {
    have "$1" || die "required command '$1' not found on PATH. ${2:-Install it and re-run.}"
}

# Venv python. Fails loudly rather than silently using the system interpreter.
venv_python() {
    local py="${VENV_DIR}/bin/python"
    [[ -x "${py}" ]] || die "virtualenv missing at ${VENV_DIR}. Run ./deploy.sh first."
    printf '%s' "${py}"
}

venv_bin() {
    local exe="${VENV_DIR}/bin/$1"
    [[ -x "${exe}" ]] || die "'$1' not found in ${VENV_DIR}/bin. Run ./deploy.sh first."
    printf '%s' "${exe}"
}

# --- Supervisor -------------------------------------------------------------
supervisor_running() {
    [[ -S "${SUPERVISOR_SOCK}" ]] || return 1
    "$(supervisorctl_bin)" -c "${SUPERVISOR_CONF}" pid >/dev/null 2>&1
}

supervisorctl_bin() {
    if [[ -x "${VENV_DIR}/bin/supervisorctl" ]]; then
        printf '%s' "${VENV_DIR}/bin/supervisorctl"
    elif have supervisorctl; then
        command -v supervisorctl
    else
        die "supervisorctl not found. Run ./deploy.sh to install it into ${VENV_DIR}."
    fi
}

supervisord_bin() {
    if [[ -x "${VENV_DIR}/bin/supervisord" ]]; then
        printf '%s' "${VENV_DIR}/bin/supervisord"
    elif have supervisord; then
        command -v supervisord
    else
        die "supervisord not found. Run ./deploy.sh to install it into ${VENV_DIR}."
    fi
}

# supervisorctl with the generated config already applied.
sctl() {
    "$(supervisorctl_bin)" -c "${SUPERVISOR_CONF}" "$@"
}

# --- HTTP -------------------------------------------------------------------
# Bounded curl. Never blocks a script forever on a hung socket.
http_get() {
    local url="$1" timeout="${2:-5}"
    curl --silent --show-error --fail \
         --connect-timeout 3 --max-time "${timeout}" \
         "${url}"
}

http_status() {
    local url="$1" timeout="${2:-5}"
    curl --silent --output /dev/null --write-out '%{http_code}' \
         --connect-timeout 3 --max-time "${timeout}" \
         "${url}" 2>/dev/null || printf '000'
}

# Poll a URL until it returns 2xx or the attempt budget is exhausted.
# wait_for_http <url> <attempts> <sleep_seconds>
wait_for_http() {
    local url="$1" attempts="${2:-30}" delay="${3:-2}" i status
    for ((i = 1; i <= attempts; i++)); do
        status="$(http_status "${url}" 5)"
        if [[ "${status}" =~ ^2 ]]; then
            log_debug "${url} -> ${status} after ${i} attempt(s)"
            return 0
        fi
        [[ "${i}" -lt "${attempts}" ]] && sleep "${delay}"
    done
    log_error "timed out waiting for ${url} (last status ${status}, ${attempts} attempts)"
    return 1
}

# --- Datastore probes -------------------------------------------------------
# Parse host/port out of redis://[:pass@]host:port/db without leaking the
# password into logs or process listings.
redis_host() {
    local url="${REDIS_URL:-redis://localhost:6379/0}" hostport
    hostport="${url#*://}"
    hostport="${hostport##*@}"
    hostport="${hostport%%/*}"
    printf '%s' "${hostport%%:*}"
}

redis_port() {
    local url="${REDIS_URL:-redis://localhost:6379/0}" hostport
    hostport="${url#*://}"
    hostport="${hostport##*@}"
    hostport="${hostport%%/*}"
    if [[ "${hostport}" == *:* ]]; then printf '%s' "${hostport##*:}"; else printf '6379'; fi
}

pg_host() {
    local url="${DATABASE_URL:-postgresql+asyncpg://orphus:orphus@localhost:5432/orphus}" rest
    rest="${url#*://}"
    rest="${rest##*@}"
    rest="${rest%%/*}"
    printf '%s' "${rest%%:*}"
}

pg_port() {
    local url="${DATABASE_URL:-postgresql+asyncpg://orphus:orphus@localhost:5432/orphus}" rest
    rest="${url#*://}"
    rest="${rest##*@}"
    rest="${rest%%/*}"
    if [[ "${rest}" == *:* ]]; then printf '%s' "${rest##*:}"; else printf '5432'; fi
}

pg_database() {
    local url="${DATABASE_URL:-postgresql+asyncpg://orphus:orphus@localhost:5432/orphus}" rest
    rest="${url##*/}"
    printf '%s' "${rest%%\?*}"
}

pg_user() {
    local url="${DATABASE_URL:-postgresql+asyncpg://orphus:orphus@localhost:5432/orphus}" creds
    creds="${url#*://}"
    if [[ "${creds}" == *@* ]]; then
        creds="${creds%%@*}"
        printf '%s' "${creds%%:*}"
    else
        printf 'orphus'
    fi
}

# TCP reachability with a bounded timeout, no extra dependencies.
tcp_open() {
    local host="$1" port="$2" timeout="${3:-3}"
    timeout "${timeout}" bash -c "exec 3<>/dev/tcp/${host}/${port}" 2>/dev/null
}

# --- GPU --------------------------------------------------------------------
gpu_present() { have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; }

# Total / used VRAM in MiB for the visible devices, one line per GPU.
gpu_memory_report() {
    gpu_present || return 1
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader
}

# --- Misc -------------------------------------------------------------------
ensure_dir() {
    local d
    for d in "$@"; do
        [[ -d "${d}" ]] || mkdir -p "${d}"
    done
}

# Human-friendly boolean for status output.
mark() { if [[ "$1" == "0" ]]; then printf '%sOK%s' "${C_GREEN}" "${C_RESET}"; else printf '%sDOWN%s' "${C_RED}" "${C_RESET}"; fi; }

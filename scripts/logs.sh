#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
load_orphus_env
service="${1:-api}"
shift || true
case "${service}" in
    api) file="${LOG_DIR}/api.log" ;;
    redis) file="${LOG_DIR}/redis.log" ;;
    postgres) file="${LOG_DIR}/postgres.log" ;;
    *) die "unknown service: ${service}" ;;
esac
lines=100
follow=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--follow) follow=1 ;;
        --tail) shift; lines="${1:?--tail requires a count}" ;;
        *) die "unknown argument: $1" ;;
    esac
    shift
done
[[ -f "${file}" ]] || die "log does not exist: ${file}"
if [[ "${follow}" -eq 1 ]]; then tail -n "${lines}" -f "${file}"; else tail -n "${lines}" "${file}"; fi


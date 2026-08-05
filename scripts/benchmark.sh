#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
load_orphus_env
py="$(venv_python)"
exec "${py}" -m pytest benchmarks -m "slow or gpu" "$@"


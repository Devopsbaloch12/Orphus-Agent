#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
load_orphus_env
http_get "${BASE_URL}/health" 15
printf '\n'


#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
load_orphus_env
"${ORPHUS_ROOT}/scripts/stop.sh" || true
find "${ORPHUS_ROOT}" -type d -name __pycache__ -prune -exec rm -rf -- {} +
find "${LOG_DIR}" -type f -name '*.log.*' -mtime +14 -delete 2>/dev/null || true
log_ok "stopped services and removed caches/rotated logs; models and databases preserved"


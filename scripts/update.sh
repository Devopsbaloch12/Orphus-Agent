#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
load_orphus_env
require_cmd git
git -C "${ORPHUS_ROOT}" pull --ff-only
"$(venv_python)" -m pip install -e "${ORPHUS_ROOT}[gpu,observability]"
"${VENV_DIR}/bin/alembic" -c "${ORPHUS_ROOT}/alembic.ini" upgrade head
exec "${ORPHUS_ROOT}/scripts/restart.sh"


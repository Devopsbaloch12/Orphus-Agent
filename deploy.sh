#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# deploy.sh -- provision and launch Orphus on a GPU host (RunPod, VM, bare metal).
#
# Purpose:
#   One idempotent command that takes a bare Ubuntu GPU host to a serving
#   Orphus instance. Running it five times in a row is safe: every step checks
#   the world before changing it, and re-running only does the work that is
#   actually outstanding.
#
# Usage:
#   ./deploy.sh [options]
#   ./deploy.sh --help
#
# Docker is deliberately NOT on this path. The production target is a RunPod
# pod that already *is* a container; nesting another runtime buys nothing and
# costs GPU passthrough complexity. docker/docker-compose.dev.yml exists only
# to give local developers Redis and PostgreSQL, and is optional.
#
# Nothing here hardcodes a secret, an API key, an absolute path, or a GPU id.
# Every such value is read from the environment with a documented default.
# ---------------------------------------------------------------------------
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/scripts/_common.sh"
install_err_trap "deploy.sh"

# --- Defaults (all overridable from the environment) ------------------------
SKIP_MODELS="${ORPHUS_SKIP_MODELS:-0}"
SKIP_DEPS="${ORPHUS_SKIP_DEPS:-0}"
SKIP_MIGRATIONS="${ORPHUS_SKIP_MIGRATIONS:-0}"
DEV_MODE=0
NO_START=0
FORCE_VENV=0
FORCE_DEPS=0
FORCE_MODELS=0
WITH_VENDOR=0
HEALTH_TIMEOUT_S="${ORPHUS_HEALTH_TIMEOUT_S:-300}"
PYTHON_BIN="${PYTHON_BIN:-}"
CUDA_CHANNEL="${TORCH_CUDA_CHANNEL:-}"
# Minimum VRAM for the documented 20-session target, in MiB.
MIN_VRAM_MIB="${ORPHUS_MIN_VRAM_MIB:-40000}"

usage() {
    cat <<'EOF'
deploy.sh -- provision and launch Orphus.

USAGE
    ./deploy.sh [options]

OPTIONS
    --dev                 Developer install: no GPU required, installs the
                          [dev,audio] extras instead of [gpu,observability],
                          and does not enforce production-only env vars.
    --skip-deps           Skip OS packages and pip installs. Use when only the
                          configuration or the model set has changed.
    --skip-models         Skip model downloads. Use when $MODEL_ROOT is already
                          populated (e.g. a RunPod network volume).
    --skip-migrations     Skip `alembic upgrade head`.
    --no-start            Provision everything but do not launch services.
    --force-venv          Delete and recreate the virtualenv.
    --force-deps          Reinstall Python dependencies even if unchanged.
    --force-models        Re-download models even if the sentinels are present.
    --with-vendor         Clone the upstream reference repos into vendor/.
    --cuda-channel <c>    Override the detected PyTorch CUDA wheel channel
                          (cu121 | cu124 | cu126 | cu128).
    --python <path>       Interpreter used to build the virtualenv.
    --health-timeout <s>  Seconds to wait for the API to become healthy
                          (default 300; a cold vLLM load is slow).
    -h, --help            Show this message.

ENVIRONMENT
    Read from .env (and .env.secrets) in the repo root; see .env.example.
    Additional knobs honoured here:

    ORPHUS_VENV           Virtualenv location            (default ./.venv)
    ORPHUS_RUN_DIR        PID/socket/state directory     (default ./run)
    MODEL_ROOT            Model weight root              (default ./models)
    LOG_DIR               Log directory                  (default ./logs)
    ORPHUS_API_COMMAND    Full command line for the API process
    ORPHUS_MANAGE_REDIS   auto | yes | no                (default auto)
    ORPHUS_MANAGE_POSTGRES auto | yes | no               (default auto)
    HF_TOKEN              Hugging Face token for gated repositories
    ORPHUS_DEBUG          Any non-empty value enables verbose tracing

EXAMPLES
    ./deploy.sh                              # full production deploy
    ./deploy.sh --skip-models                # redeploy code, keep weights
    ./deploy.sh --dev --skip-models          # laptop, no GPU
    ./deploy.sh --no-start && ./scripts/start.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)             DEV_MODE=1 ;;
        --skip-deps)       SKIP_DEPS=1 ;;
        --skip-models)     SKIP_MODELS=1 ;;
        --skip-migrations) SKIP_MIGRATIONS=1 ;;
        --no-start)        NO_START=1 ;;
        --force-venv)      FORCE_VENV=1 ;;
        --force-deps)      FORCE_DEPS=1 ;;
        --force-models)    FORCE_MODELS=1 ;;
        --with-vendor)     WITH_VENDOR=1 ;;
        --cuda-channel)    CUDA_CHANNEL="${2:?--cuda-channel needs a value}"; shift ;;
        --python)          PYTHON_BIN="${2:?--python needs a value}"; shift ;;
        --health-timeout)  HEALTH_TIMEOUT_S="${2:?--health-timeout needs a value}"; shift ;;
        -h|--help)         usage; exit 0 ;;
        *)                 log_error "unknown option: $1"; echo; usage; exit 2 ;;
    esac
    shift
done

DEPLOY_STARTED_AT="$(date +%s)"

# Production unless told otherwise. Deliberately exported into the process
# environment and never written to .env -- see the ORPHUS_API_KEYS note in
# validate_environment() for why ORPHUS_-prefixed names must not live in .env.
if [[ "${DEV_MODE}" -eq 1 ]]; then
    export ORPHUS_ENV="${ORPHUS_ENV:-development}"
else
    export ORPHUS_ENV="${ORPHUS_ENV:-production}"
fi

# Root/sudo strategy. RunPod runs as root; a VM usually does not.
if [[ "$(id -u)" -eq 0 ]]; then
    SUDO=""
elif have sudo; then
    SUDO="sudo"
else
    SUDO=""
fi
RUN_USER="${ORPHUS_RUN_USER:-$(id -un)}"

# ---------------------------------------------------------------------------
# 0. Runtime directories
#
# Out of the spec's stated order on purpose: model downloads, the state dir
# that makes every later step idempotent, and log files all need these to
# exist first.
# ---------------------------------------------------------------------------
step_directories() {
    log_step "Runtime directories"
    load_env_file "${ORPHUS_ROOT}/.env"
    load_env_file "${ORPHUS_ROOT}/.env.secrets"
    MODEL_ROOT="${MODEL_ROOT:-${ORPHUS_ROOT}/models}"
    # Relative MODEL_ROOT/LOG_DIR are resolved against the repo root, matching
    # how the Python loader resolves the same defaults.
    [[ "${MODEL_ROOT}" == /* ]] || MODEL_ROOT="${ORPHUS_ROOT}/${MODEL_ROOT#./}"
    LOG_DIR="${LOG_DIR:-${ORPHUS_ROOT}/logs}"
    [[ "${LOG_DIR}" == /* ]] || LOG_DIR="${ORPHUS_ROOT}/${LOG_DIR#./}"
    export MODEL_ROOT LOG_DIR

    ensure_dir "${LOG_DIR}" "${RUN_DIR}" "${MODEL_ROOT}" "${STATE_DIR}" \
               "${SUPERVISOR_INCLUDE_DIR}"
    log_ok "logs=${LOG_DIR}"
    log_ok "run=${RUN_DIR}"
    log_ok "models=${MODEL_ROOT}"
}

# ---------------------------------------------------------------------------
# 1. GPU detection
# ---------------------------------------------------------------------------
step_gpu() {
    log_step "GPU detection"
    if gpu_present; then
        local line
        while IFS= read -r line; do log_ok "GPU ${line}"; done < <(gpu_memory_report)
        GPU_TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
        if [[ "${GPU_TOTAL_MIB}" -lt "${MIN_VRAM_MIB}" ]]; then
            log_warn "GPU reports ${GPU_TOTAL_MIB} MiB; the documented 20-session profile assumes ~48 GiB."
            log_warn "Lower session.max_concurrent and tts.gpu_memory_utilization, or the vLLM engine will fail to allocate its KV cache."
            log_warn "See docs/performance-tuning.md -> 'Fitting a smaller card'."
        fi
        return 0
    fi

    if [[ "${DEV_MODE}" -eq 1 ]]; then
        log_warn "No NVIDIA GPU detected. Continuing because --dev was passed."
        log_warn "ASR and TTS will not load; the API, config, and test suite still work."
        return 0
    fi

    log_error "No NVIDIA GPU detected."
    log_error "  'nvidia-smi' is $(have nvidia-smi && echo 'present but listed no devices' || echo 'not on PATH')."
    log_error ""
    log_error "  Orphus serves Nemotron ASR and a 3B Orpheus TTS model from VRAM."
    log_error "  There is no CPU fallback for the production profile."
    log_error ""
    log_error "  Fix one of:"
    log_error "    * Launch a GPU pod (48 GB class: A6000 Ada, L40S, A100 40/80, H100)."
    log_error "    * Check the container has the NVIDIA runtime: 'ls /dev/nvidia*'."
    log_error "    * Install the driver: 'apt-get install -y nvidia-utils-<version>'."
    log_error "    * Re-run with --dev for a CPU-only developer install."
    exit 1
}

# ---------------------------------------------------------------------------
# 2. CUDA detection -> PyTorch wheel channel
# ---------------------------------------------------------------------------
step_cuda() {
    log_step "CUDA detection"
    if [[ -n "${CUDA_CHANNEL}" ]]; then
        log_info "CUDA wheel channel pinned by flag/env: ${CUDA_CHANNEL}"
        return 0
    fi
    if [[ "${DEV_MODE}" -eq 1 ]] && ! gpu_present; then
        CUDA_CHANNEL="cpu"
        log_info "No GPU; using the CPU PyTorch channel."
        return 0
    fi

    local driver cuda_ver major minor
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
    # The driver's maximum supported CUDA runtime, printed in the nvidia-smi banner.
    cuda_ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
    if [[ -z "${cuda_ver}" ]] && have nvcc; then
        cuda_ver="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
    fi
    [[ -n "${cuda_ver}" ]] || die "could not determine the CUDA version. Pass --cuda-channel cu124 (or the value matching your driver)."

    major="${cuda_ver%%.*}"
    minor="${cuda_ver#*.}"
    log_ok "driver ${driver}, CUDA ${cuda_ver}"

    if [[ "${major}" -lt 12 ]]; then
        log_error "CUDA ${cuda_ver} is too old."
        log_error "  vllm>=0.7.3 and torch>=2.5 both require a CUDA 12.x driver."
        log_error "  Upgrade the host driver or pick a newer RunPod base image."
        exit 1
    fi

    if   [[ "${minor}" -ge 8 ]]; then CUDA_CHANNEL="cu128"
    elif [[ "${minor}" -ge 6 ]]; then CUDA_CHANNEL="cu126"
    elif [[ "${minor}" -ge 4 ]]; then CUDA_CHANNEL="cu124"
    else                              CUDA_CHANNEL="cu121"
    fi
    log_ok "PyTorch wheel channel: ${CUDA_CHANNEL}"
}

# ---------------------------------------------------------------------------
# 3. Python
# ---------------------------------------------------------------------------
step_python() {
    log_step "Python interpreter"
    local candidates=() c ver
    [[ -n "${PYTHON_BIN}" ]] && candidates+=("${PYTHON_BIN}")
    candidates+=(python3.12 python3.13 python3 python)

    for c in "${candidates[@]}"; do
        have "$c" || continue
        ver="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
        [[ -n "${ver}" ]] || continue
        case "${ver}" in
            3.12) PYTHON_BIN="$(command -v "$c")"; log_ok "using $(command -v "$c") (${ver})"; return 0 ;;
        esac
    done

    # No exact 3.12: accept a newer interpreter but say plainly what breaks.
    for c in "${candidates[@]}"; do
        have "$c" || continue
        ver="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
        [[ -n "${ver}" ]] || continue
        if "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)'; then
            PYTHON_BIN="$(command -v "$c")"
            log_warn "Python ${ver} satisfies requires-python>=3.12, but the GPU stack"
            log_warn "(vllm, nemo_toolkit, onnxruntime-gpu) publishes cp312 wheels first."
            log_warn "If pip starts building from source, install python3.12 and re-run with --python."
            return 0
        fi
    done

    log_error "No Python >= 3.12 found (tried: ${candidates[*]})."
    log_error "  Ubuntu 22.04/24.04:"
    log_error "    ${SUDO:+sudo }add-apt-repository -y ppa:deadsnakes/ppa"
    log_error "    ${SUDO:+sudo }apt-get update"
    log_error "    ${SUDO:+sudo }apt-get install -y python3.12 python3.12-venv python3.12-dev"
    log_error "  Then re-run:  ./deploy.sh --python \$(command -v python3.12)"
    exit 1
}

# ---------------------------------------------------------------------------
# 4. OS packages
# ---------------------------------------------------------------------------
apt_installed() {
    dpkg-query -W -f='${Status}' "$1" 2>/dev/null | grep -q 'ok installed'
}

step_os_packages() {
    log_step "OS packages"
    if [[ "${SKIP_DEPS}" -eq 1 ]]; then log_info "skipped (--skip-deps)"; return 0; fi

    if ! have apt-get; then
        log_warn "apt-get not found; skipping OS package installation."
        log_warn "Ensure these are present by other means: libsndfile1 ffmpeg build-essential pkg-config git curl."
        return 0
    fi

    # libsndfile1 -> soundfile;  ffmpeg -> librosa/audio decode;
    # build-essential + pkg-config -> source builds when a wheel is missing;
    # ninja-build -> vLLM/flashinfer JIT-compile attention kernels on first
    # start. Without it the engine loads the weights, then dies with
    # FileNotFoundError: 'ninja' and takes the whole API down with it.
    local want=(libsndfile1 ffmpeg build-essential pkg-config git curl ca-certificates procps ninja-build)
    # Only pull in a server if this host is going to run one itself.
    [[ "$(manage_redis_decision)"    == "yes" ]] && want+=(redis-server redis-tools)
    [[ "$(manage_postgres_decision)" == "yes" ]] && want+=(postgresql postgresql-client)

    local missing=() p
    for p in "${want[@]}"; do
        apt_installed "$p" || missing+=("$p")
    done

    if [[ "${#missing[@]}" -eq 0 ]]; then
        log_ok "all ${#want[@]} packages already installed"
        return 0
    fi

    log_info "installing: ${missing[*]}"
    if [[ -z "${SUDO}" && "$(id -u)" -ne 0 ]]; then
        log_error "Missing OS packages and no way to install them (not root, no sudo)."
        log_error "  Ask an administrator to run:"
        log_error "    apt-get update && apt-get install -y ${missing[*]}"
        exit 1
    fi

    # Refresh the package index at most once a day; a full apt-get update on
    # every deploy is the single slowest idempotent-but-wasteful step.
    local stamp="${STATE_DIR}/apt-updated"
    if [[ ! -f "${stamp}" ]] || [[ -n "$(find "${stamp}" -mmin +1440 2>/dev/null)" ]]; then
        DEBIAN_FRONTEND=noninteractive ${SUDO} apt-get update -qq
        : > "${stamp}"
    fi
    DEBIAN_FRONTEND=noninteractive ${SUDO} apt-get install -y --no-install-recommends "${missing[@]}"
    log_ok "installed ${#missing[@]} package(s)"
}

# ---------------------------------------------------------------------------
# 5. Virtualenv
# ---------------------------------------------------------------------------
step_venv() {
    log_step "Virtualenv"
    if [[ "${FORCE_VENV}" -eq 1 && -d "${VENV_DIR}" ]]; then
        log_warn "removing ${VENV_DIR} (--force-venv)"
        rm -rf "${VENV_DIR}"
    fi

    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        local have_ver want_ver
        have_ver="$("${VENV_DIR}/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        want_ver="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        if [[ "${have_ver}" == "${want_ver}" ]]; then
            log_ok "reusing ${VENV_DIR} (Python ${have_ver})"
            return 0
        fi
        log_warn "venv is Python ${have_ver} but ${PYTHON_BIN} is ${want_ver}; recreating"
        rm -rf "${VENV_DIR}"
    fi

    log_info "creating ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}" \
        || die "venv creation failed. On Debian/Ubuntu install the matching venv package: ${SUDO:+sudo }apt-get install -y python3.12-venv"
    log_ok "created"
}

# ---------------------------------------------------------------------------
# 6. Python dependencies
# ---------------------------------------------------------------------------
dep_fingerprint() {
    # Any input that changes the resolved environment invalidates the marker.
    {
        cat "${ORPHUS_ROOT}/pyproject.toml"
        printf 'extras=%s channel=%s dev=%s\n' "${PIP_EXTRAS}" "${CUDA_CHANNEL}" "${DEV_MODE}"
        "${VENV_DIR}/bin/python" -c 'import sys; print(sys.version)'
    } | sha256sum | cut -d' ' -f1
}

step_dependencies() {
    log_step "Python dependencies"
    if [[ "${DEV_MODE}" -eq 1 ]]; then
        PIP_EXTRAS="dev,audio"
    else
        PIP_EXTRAS="gpu,observability"
    fi

    if [[ "${SKIP_DEPS}" -eq 1 ]]; then
        log_info "skipped (--skip-deps)"
        return 0
    fi

    local marker="${STATE_DIR}/deps.sha256" fingerprint
    fingerprint="$(dep_fingerprint)"
    if [[ "${FORCE_DEPS}" -eq 0 && -f "${marker}" && "$(cat "${marker}")" == "${fingerprint}" ]]; then
        log_ok "dependencies unchanged since the last deploy; skipping"
        return 0
    fi

    local pip="${VENV_DIR}/bin/pip"
    "${pip}" install --quiet --upgrade pip setuptools wheel

    # PyTorch first, straight from the channel that matches this driver. Doing
    # it before the extras means the later resolve sees torch>=2.5 already
    # satisfied and leaves the CUDA build alone.
    if [[ "${CUDA_CHANNEL}" == "cpu" ]]; then
        log_info "installing CPU PyTorch"
        "${pip}" install --index-url "https://download.pytorch.org/whl/cpu" torch torchaudio
    else
        log_info "installing PyTorch (${CUDA_CHANNEL})"
        "${pip}" install --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}" torch torchaudio
    fi

    log_info "installing orphus[${PIP_EXTRAS}]"
    "${pip}" install -e "${ORPHUS_ROOT}[${PIP_EXTRAS}]"

    # Deploy-time tooling. Deliberately not in pyproject.toml: supervisor and
    # the HF CLI are properties of *this* deployment, not of the package.
    log_info "installing deploy tooling (supervisor, huggingface_hub CLI, silero-vad)"
    # silero-vad is imported by orphus.vad.silero at model-load time but is not
    # pulled in by any extra in pyproject.toml; without it the API dies at
    # startup with ModuleNotFoundError: No module named 'silero_vad'.
    "${pip}" install --quiet "supervisor>=4.2" "huggingface_hub[cli,hf_transfer]>=0.26" "silero-vad"

    # The base install is GPU-free by design, so a dev machine has no ONNX
    # runtime at all and Silero VAD cannot load. Add the CPU build there.
    if [[ "${DEV_MODE}" -eq 1 ]]; then
        log_info "installing onnxruntime (CPU) so Silero VAD can load without a GPU"
        "${pip}" install --quiet "onnxruntime>=1.20"
    fi

    printf '%s' "${fingerprint}" > "${marker}"
    log_ok "dependencies installed"

    check_dependency_conflicts
}

check_dependency_conflicts() {
    # `pip check` catches the realistic failure here: vllm, nemo_toolkit and
    # transformers>=5.13 all pin transformers, and their ranges do not
    # obviously intersect. Surface it now rather than at model load.
    local out
    if out="$("${VENV_DIR}/bin/pip" check 2>&1)"; then
        log_ok "pip check: no dependency conflicts"
    else
        log_warn "pip check reported conflicts:"
        printf '%s\n' "${out}" | sed 's/^/         /' >&2
        log_warn "The ASR stack can run under either nemo_toolkit or transformers>=5.13."
        log_warn "If they cannot co-install, drop one -- see docs/troubleshooting.md."
    fi
}

# ---------------------------------------------------------------------------
# 7. Verify PyTorch sees CUDA
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 6b. Third-party patch: orpheus_tts.OrpheusModel
#
# The published orpheus-speech wheel predates the API this codebase targets --
# its __init__ takes only (model_name, dtype), so orphus.tts.orpheus.load()
# raises TypeError: unexpected keyword argument 'tokenizer' and the API can
# never start. Widen the signature in place: accept `tokenizer` and pass any
# further kwargs (max_model_len, gpu_memory_utilization) through to
# AsyncEngineArgs. Idempotent -- re-running is a no-op once patched.
# ---------------------------------------------------------------------------
step_patch_orpheus() {
    log_step "Third-party patches"
    local target
    target="$("${VENV_DIR}/bin/python" - <<'PY' 2>/dev/null || true
try:
    import orpheus_tts.engine_class as m
    print(m.__file__)
except Exception:
    pass
PY
)"
    if [[ -z "${target}" || ! -f "${target}" ]]; then
        log_warn "orpheus_tts not importable; skipping patch"
        return 0
    fi
    if grep -q "engine_kwargs" "${target}"; then
        log_ok "orpheus_tts already patched"
        return 0
    fi
    cp -f "${target}" "${target}.orig"
    "${VENV_DIR}/bin/python" - "${target}" <<'PY'
import re, sys
path = sys.argv[1]
src = open(path).read()
src = src.replace(
    "def __init__(self, model_name, dtype=torch.bfloat16):",
    "def __init__(self, model_name, dtype=torch.bfloat16, tokenizer=None, **engine_kwargs):",
    1,
)
src = src.replace(
    "self.dtype = dtype\n        self.engine = self._setup_engine()",
    "self.dtype = dtype\n        self.engine_kwargs = engine_kwargs\n        self.engine = self._setup_engine()",
    1,
)
src = src.replace(
    "self.tokeniser = AutoTokenizer.from_pretrained(model_name)",
    "self.tokeniser = AutoTokenizer.from_pretrained(tokenizer or model_name)",
    1,
)
src = re.sub(
    r"(engine_args = AsyncEngineArgs\(\s*\n\s*model=self\.model_name,\s*\n\s*dtype=self\.dtype,\s*\n)(\s*\))",
    r"\1            **self.engine_kwargs,\n\2",
    src,
    count=1,
)
open(path, "w").write(src)
PY
    if grep -q "engine_kwargs" "${target}"; then
        log_ok "patched $(basename "${target}") (original kept as .orig)"
    else
        cp -f "${target}.orig" "${target}"
        die "failed to patch orpheus_tts; restored the original. Upstream layout may have changed."
    fi

    patch_nemotron_generate
}

# transformers' Nemotron3_5AsrGenerationMixin.generate() stashes state on the
# shared model instance and deletes it in a finally block. One model serves
# every session here, so concurrent transcriptions race: thread A's cleanup
# makes thread B fail with
#   AttributeError: 'Nemotron3_5AsrForRNNT' object has no attribute 'get_audio_features'
# and the caller gets silence. The stashing exists only to inject prompt_ids,
# which this pipeline never sends, so take a fast path when there is no prompt.
patch_nemotron_generate() {
    local gen
    gen="$("${VENV_DIR}/bin/python" - <<'PY' 2>/dev/null || true
try:
    import transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr as m
    print(m.__file__)
except Exception:
    pass
PY
)"
    if [[ -z "${gen}" || ! -f "${gen}" ]]; then
        log_warn "nemotron3_5_asr generation module not found; skipping patch"
        return 0
    fi
    if grep -q "Fast path, and the only thread-safe one" "${gen}"; then
        log_ok "nemotron ASR generate already patched"
        return 0
    fi
    cp -f "${gen}" "${gen}.orig"
    "${VENV_DIR}/bin/python" - "${gen}" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
old = """        self._prompt_ids = kwargs.pop("prompt_ids", None)
        get_audio_features = self.get_audio_features"""
new = """        self._prompt_ids = kwargs.pop("prompt_ids", None)
        if self._prompt_ids is None:
            # Fast path, and the only thread-safe one. The wrapper below exists
            # solely to inject prompt_ids into get_audio_features; with no
            # prompt it is a no-op, while its instance-attribute juggling is a
            # data race across concurrent sessions on this shared model.
            del self._prompt_ids
            return super().generate(inputs=inputs, generation_config=generation_config, **kwargs)
        get_audio_features = self.get_audio_features"""
if old not in src:
    sys.exit("anchor not found")
open(path, "w").write(src.replace(old, new, 1))
PY
    if grep -q "Fast path, and the only thread-safe one" "${gen}"; then
        log_ok "patched $(basename "${gen}") (original kept as .orig)"
    else
        cp -f "${gen}.orig" "${gen}"
        die "failed to patch nemotron generate; restored the original."
    fi
}

step_verify_torch() {
    log_step "PyTorch / CUDA verification"
    if [[ "${SKIP_DEPS}" -eq 1 && ! -x "${VENV_DIR}/bin/python" ]]; then
        log_info "skipped"; return 0
    fi
    if [[ "${DEV_MODE}" -eq 1 ]] && ! gpu_present; then
        log_info "skipped (--dev, no GPU)"; return 0
    fi

    local out
    if out="$("${VENV_DIR}/bin/python" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("UNAVAILABLE", torch.__version__, torch.version.cuda, sep="|")
    sys.exit(3)

i = torch.cuda.current_device()
props = torch.cuda.get_device_properties(i)
print(
    "OK",
    torch.__version__,
    torch.version.cuda,
    props.name,
    f"{props.total_memory // (1024 * 1024)}MiB",
    f"sm_{props.major}{props.minor}",
    sep="|",
)
PY
    )"; then
        IFS='|' read -r _ tv cv name mem sm <<< "${out}"
        log_ok "torch ${tv} (CUDA ${cv}) sees ${name}, ${mem}, ${sm}"
    else
        IFS='|' read -r _ tv cv <<< "${out}"
        log_error "PyTorch ${tv:-?} cannot see CUDA (built against ${cv:-none})."
        log_error "  Most common causes:"
        log_error "    * A CPU-only wheel got installed. Re-run: ./deploy.sh --force-deps"
        log_error "    * The container has no GPU passthrough: 'ls -l /dev/nvidia*'"
        log_error "    * Driver/runtime mismatch. Detected channel was ${CUDA_CHANNEL};"
        log_error "      override with: ./deploy.sh --cuda-channel cu124 --force-deps"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 8. Models
# ---------------------------------------------------------------------------
hf_cli() {
    if   [[ -x "${VENV_DIR}/bin/hf" ]];              then printf '%s' "${VENV_DIR}/bin/hf"
    elif [[ -x "${VENV_DIR}/bin/huggingface-cli" ]]; then printf '%s' "${VENV_DIR}/bin/huggingface-cli"
    else die "no Hugging Face CLI in ${VENV_DIR}. Re-run without --skip-deps."
    fi
}

download_hf_repo() {
    local repo="$1" dest="$2" label="$3"; shift 3
    local sentinel="${dest}/.orphus-complete"

    if [[ "${FORCE_MODELS}" -eq 0 && -f "${sentinel}" ]] \
       && grep -qxF "${repo}" "${sentinel}" 2>/dev/null; then
        log_ok "${label}: already present (${dest})"
        return 0
    fi

    ensure_dir "${dest}"
    log_info "${label}: downloading ${repo}"
    local cli out rc=0
    cli="$(hf_cli)"
    # hf_transfer gives a large speedup on the multi-GB TTS checkpoint.
    # download_timeout_s from config/default.yaml bounds the whole transfer.
    if ! out="$(HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
                timeout "${MODEL_DOWNLOAD_TIMEOUT_S}" \
                "${cli}" download "${repo}" --local-dir "${dest}" "$@" 2>&1)"; then
        rc=$?
        printf '%s\n' "${out}" | tail -20 | sed 's/^/         /' >&2
        if [[ "${rc}" -eq 124 ]]; then
            log_error "${label}: download timed out after ${MODEL_DOWNLOAD_TIMEOUT_S}s."
            log_error "  Raise models.download_timeout_s in config/default.yaml or"
            log_error "  export ORPHUS_MODEL_DOWNLOAD_TIMEOUT_S=3600 and re-run."
        elif printf '%s' "${out}" | grep -qiE '401|403|gated|awaiting a review|access to model'; then
            log_error "${label}: ${repo} is gated or requires authentication."
            log_error "  1. Accept the licence at https://huggingface.co/${repo}"
            log_error "  2. export HF_TOKEN=hf_xxx   (a read token)"
            log_error "  3. ./deploy.sh --skip-deps"
        else
            log_error "${label}: download of ${repo} failed."
        fi
        exit 1
    fi
    printf '%s\n' "${repo}" > "${sentinel}"
    log_ok "${label}: ${dest}"
}

download_silero_vad() {
    local dest="${MODEL_ROOT}/vad"
    local target="${dest}/silero_vad.onnx"
    local vendored="${ORPHUS_ROOT}/vendor/silero-vad/src/silero_vad/data/silero_vad.onnx"
    local ref="${SILERO_VAD_REF:-master}"

    if [[ "${FORCE_MODELS}" -eq 0 && -s "${target}" ]]; then
        log_ok "VAD: already present (${target})"
        return 0
    fi
    ensure_dir "${dest}"

    if [[ -s "${vendored}" ]]; then
        cp -f "${vendored}" "${target}"
        log_ok "VAD: copied from vendor/silero-vad"
    else
        log_info "VAD: downloading silero_vad.onnx (${ref})"
        curl -fsSL --retry 3 --retry-delay 2 --max-time 120 \
            "https://raw.githubusercontent.com/snakers4/silero-vad/${ref}/src/silero_vad/data/silero_vad.onnx" \
            -o "${target}" \
            || die "VAD download failed. Check outbound HTTPS to raw.githubusercontent.com, or set SILERO_VAD_REF to a tag."
    fi

    # The v5 ONNX graph is ~2.2 MB. A truncated or HTML error page is far
    # smaller and would fail much later, inside onnxruntime.
    local size
    size="$(stat -c %s "${target}" 2>/dev/null || echo 0)"
    [[ "${size}" -gt 1000000 ]] || die "VAD model at ${target} is only ${size} bytes; the download was truncated. Delete it and re-run."
    printf 'snakers4/silero-vad@%s\n' "${ref}" > "${dest}/.orphus-complete"
    log_ok "VAD: ${target} (${size} bytes)"
}

step_models() {
    log_step "Models"
    MODEL_DOWNLOAD_TIMEOUT_S="${ORPHUS_MODEL_DOWNLOAD_TIMEOUT_S:-1800}"

    if [[ "${SKIP_MODELS}" -eq 1 ]]; then
        log_info "skipped (--skip-models)"
        return 0
    fi

    local asr_repo="${ASR_MODEL_ID:-nvidia/nemotron-3.5-asr-streaming-0.6b}"
    local tts_repo="${TTS_MODEL_ID:-canopylabs/orpheus-tts-0.1-finetune-prod}"
    local snac_repo="${SNAC_MODEL_ID:-hubertsiuzdak/snac_24khz}"
    local tok_repo="${ORPHEUS_TOKENIZER_ID:-canopylabs/orpheus-3b-0.1-pretrained}"

    # 1/5 VAD -- Silero, ~2 MB ONNX, CPU.
    download_silero_vad

    # 2/5 ASR -- Nemotron 3.5 cache-aware streaming FastConformer-RNNT, 638M.
    #     Licence OpenMDW-1.1; accept it on the model page before deploying.
    download_hf_repo "${asr_repo}" "${MODEL_ROOT}/asr" "ASR"

    # 3/5 TTS -- Orpheus 3B. The 1B/small variants referenced in older specs
    #     were never released; only medium-3b exists upstream.
    download_hf_repo "${tts_repo}" "${MODEL_ROOT}/tts" "TTS"

    # 4/5 SNAC -- the neural codec that turns Orpheus tokens into 24 kHz PCM.
    #     Without it the TTS path produces tokens and no audio.
    download_hf_repo "${snac_repo}" "${MODEL_ROOT}/snac" "SNAC"

    # 5/5 Orpheus tokenizer. Upstream OrpheusModel defaults its tokenizer to the
    #     *pretrained* repo, not the finetune-prod one, so a fully offline node
    #     needs it cached locally. Tokenizer files only, a few MB.
    download_hf_repo "${tok_repo}" "${MODEL_ROOT}/tts-tokenizer" "TTS tokenizer" \
        --include "tokenizer*" "special_tokens_map.json" "*.txt" "config.json"

    # Postcondition. huggingface_hub 1.x consumes only the first --include
    # pattern, so the download above can silently land config.json and
    # special_tokens_map.json without tokenizer.json -- and the API then dies
    # at startup with "Couldn't instantiate the backend tokenizer". The
    # finetune-prod TTS repo carries identical tokenizer files, so fall back
    # to those rather than failing the deploy.
    if [[ ! -s "${MODEL_ROOT}/tts-tokenizer/tokenizer.json" ]]; then
        log_warn "TTS tokenizer: tokenizer.json missing; copying from ${MODEL_ROOT}/tts"
        local f copied=0
        for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do
            if [[ -s "${MODEL_ROOT}/tts/${f}" ]]; then
                cp -f "${MODEL_ROOT}/tts/${f}" "${MODEL_ROOT}/tts-tokenizer/${f}" && copied=$((copied + 1))
            fi
        done
        [[ -s "${MODEL_ROOT}/tts-tokenizer/tokenizer.json" ]] \
            || die "TTS tokenizer incomplete and ${MODEL_ROOT}/tts has no tokenizer.json to copy."
        log_ok "TTS tokenizer: repaired (${copied} file(s) copied)"
    fi

    log_info "model root layout:"
    du -sh "${MODEL_ROOT}"/* 2>/dev/null | sed 's/^/         /' || true
}

step_vendor() {
    [[ "${WITH_VENDOR}" -eq 1 ]] || return 0
    log_step "Vendored reference repositories"
    require_cmd git
    ensure_dir "${ORPHUS_ROOT}/vendor"
    local spec name url
    for spec in \
        "silero-vad=https://github.com/snakers4/silero-vad" \
        "orpheus-tts=https://github.com/canopyai/Orpheus-TTS"
    do
        name="${spec%%=*}"; url="${spec#*=}"
        if [[ -d "${ORPHUS_ROOT}/vendor/${name}/.git" ]]; then
            log_ok "vendor/${name}: present"
        else
            log_info "cloning ${url}"
            git clone --depth 1 "${url}" "${ORPHUS_ROOT}/vendor/${name}"
        fi
    done
    log_info "vendor/ is gitignored and is reference material only; nothing imports from it."
}

# ---------------------------------------------------------------------------
# 9. Configuration from templates
# ---------------------------------------------------------------------------
# render_template <src> <dst> KEY=VALUE...
# Returns 0 when the destination changed, 1 when it was already correct. That
# distinction is what lets the supervisor step avoid pointless restarts.
render_template() {
    local src="$1" dst="$2"; shift 2
    [[ -f "${src}" ]] || die "missing template ${src}"
    local tmp pair k v
    tmp="$(mktemp "${STATE_DIR}/render.XXXXXX")"
    cp "${src}" "${tmp}"
    for pair in "$@"; do
        k="${pair%%=*}"; v="${pair#*=}"
        v="${v//\\/\\\\}"; v="${v//|/\\|}"; v="${v//&/\\&}"
        sed -i "s|{{${k}}}|${v}|g" "${tmp}"
    done
    if grep -q '{{[A-Z_]*}}' "${tmp}"; then
        log_warn "unsubstituted placeholders in ${dst}: $(grep -o '{{[A-Z_]*}}' "${tmp}" | sort -u | tr '\n' ' ')"
    fi
    if [[ -f "${dst}" ]] && cmp -s "${tmp}" "${dst}"; then
        rm -f "${tmp}"
        return 1
    fi
    mv "${tmp}" "${dst}"
    chmod 0644 "${dst}"
    return 0
}

step_config() {
    log_step "Configuration"

    if [[ ! -f "${ORPHUS_ROOT}/.env" ]]; then
        [[ -f "${ORPHUS_ROOT}/.env.example" ]] || die "neither .env nor .env.example exists; cannot generate configuration."
        cp "${ORPHUS_ROOT}/.env.example" "${ORPHUS_ROOT}/.env"
        chmod 0600 "${ORPHUS_ROOT}/.env"
        log_ok "generated .env from .env.example (mode 0600)"
        GENERATED_ENV=1
    else
        log_ok ".env present"
        # A world-readable secrets file is a finding, not a style preference.
        local mode
        mode="$(stat -c %a "${ORPHUS_ROOT}/.env" 2>/dev/null || echo '')"
        if [[ -n "${mode}" && "${mode}" != "600" && "${mode}" != "400" ]]; then
            chmod 0600 "${ORPHUS_ROOT}/.env"
            log_warn ".env was mode ${mode}; tightened to 0600"
        fi
    fi

    if [[ ! -f "${ORPHUS_ROOT}/.env.secrets" ]]; then
        cat > "${ORPHUS_ROOT}/.env.secrets" <<'EOF'
# ---------------------------------------------------------------------------
# Orphus -- environment-only settings. Generated by deploy.sh; gitignored.
#
# Variables that begin with ORPHUS_ but are NOT top-level Settings fields must
# live HERE and not in .env. pydantic-settings parses .env with extra="forbid",
# so an ORPHUS_-prefixed key it cannot map to a field raises a SettingsError at
# import time. This file is exported into the process environment by
# scripts/start.sh, where FlatEnvAliasSource picks it up correctly.
#
# See docs/configuration.md -> "Why ORPHUS_API_KEYS cannot live in .env".
# ---------------------------------------------------------------------------

# Comma-separated API keys required by clients. MUST be non-empty when
# ORPHUS_ENV=production; Settings refuses to start otherwise.
ORPHUS_API_KEYS=
EOF
        chmod 0600 "${ORPHUS_ROOT}/.env.secrets"
        log_ok "generated .env.secrets (mode 0600)"
    else
        log_ok ".env.secrets present"
    fi

    # Reload so validation and the supervisor render see the generated files.
    load_env_file "${ORPHUS_ROOT}/.env"
    load_env_file "${ORPHUS_ROOT}/.env.secrets"
    HOST="${HOST:-0.0.0.0}"
    PORT="${PORT:-8000}"
    PROBE_HOST="${ORPHUS_PROBE_HOST:-127.0.0.1}"
    BASE_URL="${ORPHUS_BASE_URL:-http://${PROBE_HOST}:${PORT}}"
    export HOST PORT
}

# ---------------------------------------------------------------------------
# 10. Environment validation -- collect every problem, then fail once
# ---------------------------------------------------------------------------
validate_environment() {
    log_step "Environment validation"
    local problems=()

    # --- Always required ---
    [[ -n "${REDIS_URL:-}" ]]    || problems+=("REDIS_URL is unset. Example: redis://localhost:6379/0")
    [[ -n "${DATABASE_URL:-}" ]] || problems+=("DATABASE_URL is unset. Example: postgresql+asyncpg://orphus:orphus@localhost:5432/orphus")
    [[ -n "${MODEL_ROOT:-}" ]]   || problems+=("MODEL_ROOT is unset. Example: ./models")

    if [[ -n "${DATABASE_URL:-}" && "${DATABASE_URL}" != postgresql+asyncpg://* ]]; then
        problems+=("DATABASE_URL must use the postgresql+asyncpg:// driver; SQLAlchemy is configured for the async engine. Got: ${DATABASE_URL%%://*}://")
    fi
    if [[ -n "${PORT:-}" && ! "${PORT}" =~ ^[0-9]+$ ]]; then
        problems+=("PORT must be an integer, got '${PORT}'")
    fi

    # --- Grok ---
    if [[ -z "${GROK_API_KEY:-}" ]]; then
        if [[ "${ORPHUS_ENV}" == "production" ]]; then
            problems+=("GROK_API_KEY is empty. Settings refuses to start in production without it. Get a key at https://console.x.ai and put it in .env")
        else
            log_warn "GROK_API_KEY is empty; the LLM stage will fail on first use."
        fi
    fi

    # --- The .env / ORPHUS_ prefix trap ---
    # pydantic-settings' DotEnvSettingsSource raises on any ORPHUS_-prefixed key
    # in .env that is not a top-level Settings field. ORPHUS_API_KEYS is exactly
    # that: a flat alias for security.api_keys. Empty is tolerated (the source
    # skips empty values), which is why .env.example ships it blank -- and why
    # filling it in breaks startup.
    if [[ -f "${ORPHUS_ROOT}/.env" ]] \
       && grep -qE '^[[:space:]]*(export[[:space:]]+)?ORPHUS_[A-Z0-9_]*=[^[:space:]]' "${ORPHUS_ROOT}/.env"; then
        local offenders
        offenders="$(grep -oE '^[[:space:]]*(export[[:space:]]+)?ORPHUS_[A-Z0-9_]*=[^[:space:]]' "${ORPHUS_ROOT}/.env" \
                     | sed -E 's/^[[:space:]]*(export[[:space:]]+)?//; s/=.*//' | sort -u | tr '\n' ' ')"
        # Nested-form names (ORPHUS_ASR__...) map fine; only flat aliases break.
        local bad="" name
        for name in ${offenders}; do
            [[ "${name}" == *__* ]] && continue
            bad+="${name} "
        done
        if [[ -n "${bad}" ]]; then
            problems+=("These ORPHUS_-prefixed variables are set in .env: ${bad}
             pydantic-settings parses .env with extra=\"forbid\" and cannot map them to a
             top-level Settings field, so the app raises SettingsError at startup.
             Move them to .env.secrets (already generated), which start.sh exports into
             the process environment where FlatEnvAliasSource resolves them correctly.")
        fi
    fi

    # --- Production-only invariants enforced by Settings ---
    if [[ "${ORPHUS_ENV}" == "production" && -z "${ORPHUS_API_KEYS:-}" ]]; then
        problems+=("ORPHUS_API_KEYS is empty. Settings refuses production startup with an
             unauthenticated API. Set it in .env.secrets:
                 ORPHUS_API_KEYS=\$(openssl rand -hex 32)")
    fi

    # --- Cross-section invariant that Settings enforces at startup ---
    local max_sessions batch
    max_sessions="${MAX_CONCURRENT_SESSIONS:-20}"
    batch="${ORPHUS_ASR__MAX_BATCH_SIZE:-20}"
    if [[ "${max_sessions}" =~ ^[0-9]+$ && "${batch}" =~ ^[0-9]+$ && "${max_sessions}" -gt "${batch}" ]]; then
        problems+=("MAX_CONCURRENT_SESSIONS (${max_sessions}) exceeds asr.max_batch_size (${batch}).
             Settings rejects this: a full house would not fit in one ASR batch.
             Raise it too:  ORPHUS_ASR__MAX_BATCH_SIZE=${max_sessions}")
    fi

    # --- ASR latency ladder ---
    if [[ -n "${ORPHUS_ASR__ATT_CONTEXT_SIZE:-}" ]]; then
        local right="${ORPHUS_ASR__ATT_CONTEXT_SIZE##*,}"
        right="$(printf '%s' "${right}" | tr -cd '0-9')"
        case "${right}" in
            0|1|3|6|13) : ;;
            *) problems+=("ORPHUS_ASR__ATT_CONTEXT_SIZE right context '${right}' is not a trained
             configuration. Valid: 0, 1, 3, 6, 13 -> 80/160/320/560/1120 ms lookahead.
             See docs/performance-tuning.md.") ;;
        esac
    fi

    # --- GPU id sanity ---
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "" ]]; then
        local dev="${ORPHUS_ASR__DEVICE:-cuda:0}"
        if [[ "${dev}" != "cuda:0" && "${dev}" != "cpu" ]]; then
            log_warn "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} remaps device ordinals."
            log_warn "asr/tts device should stay 'cuda:0' -- it indexes the *visible* set, not the physical bus."
        fi
    fi

    if [[ "${#problems[@]}" -gt 0 ]]; then
        log_error "Configuration is incomplete. ${#problems[@]} problem(s):"
        local i=1 p
        for p in "${problems[@]}"; do
            printf '\n  %s%d)%s %s\n' "${C_RED}" "${i}" "${C_RESET}" "${p}" >&2
            i=$((i + 1))
        done
        printf '\n' >&2
        log_error "Edit ${ORPHUS_ROOT}/.env and ${ORPHUS_ROOT}/.env.secrets, then re-run:"
        log_error "    ./deploy.sh --skip-deps --skip-models"
        exit 1
    fi
    log_ok "all required variables present and consistent"
}

# ---------------------------------------------------------------------------
# 11. Redis
# ---------------------------------------------------------------------------
is_local_host() {
    case "$1" in
        localhost|127.0.0.1|::1|0.0.0.0|"") return 0 ;;
        *) return 1 ;;
    esac
}

manage_redis_decision() {
    case "${ORPHUS_MANAGE_REDIS:-auto}" in
        yes) printf 'yes' ;;
        no)  printf 'no'  ;;
        *)
            if is_local_host "$(redis_host)" && ! tcp_open "$(redis_host)" "$(redis_port)" 2; then
                printf 'yes'
            else
                printf 'no'
            fi
            ;;
    esac
}

step_redis() {
    log_step "Redis"
    local host port
    host="$(redis_host)"; port="$(redis_port)"

    if tcp_open "${host}" "${port}" 3; then
        log_ok "already listening on ${host}:${port}"
    elif [[ "$(manage_redis_decision)" == "yes" ]]; then
        have redis-server || die "REDIS_URL points at ${host}:${port} but redis-server is not installed. Install it (${SUDO:+sudo }apt-get install -y redis-server) or point REDIS_URL at a managed instance."
        ensure_dir "${RUN_DIR}/redis"
        if render_template "${ORPHUS_ROOT}/deploy/supervisor/orphus-redis.conf.template" \
                           "${SUPERVISOR_INCLUDE_DIR}/orphus-redis.conf" \
                           "REDIS_SERVER=$(command -v redis-server)" \
                           "REDIS_PORT=${port}" \
                           "REDIS_BIND=${ORPHUS_REDIS_BIND:-127.0.0.1}" \
                           "REDIS_DIR=${RUN_DIR}/redis" \
                           "REDIS_MAXMEMORY=${ORPHUS_REDIS_MAXMEMORY:-2gb}" \
                           "RUN_DIR=${RUN_DIR}" \
                           "LOG_DIR=${LOG_DIR}" \
                           "RUN_USER=${RUN_USER}"; then
            SUPERVISOR_DIRTY=1
            log_ok "rendered supervisor program orphus-redis"
        else
            log_ok "supervisor program orphus-redis already current"
        fi
        MANAGED_REDIS=1
    else
        log_error "Redis is unreachable at ${host}:${port} and this host is not configured to manage it."
        log_error "  Either start it, or set ORPHUS_MANAGE_REDIS=yes, or point REDIS_URL elsewhere."
        exit 1
    fi
}

verify_redis() {
    # Uses the redis client from the base install so the URL (and any password
    # in it) never reaches a process argument list.
    if REDIS_URL="${REDIS_URL}" "${VENV_DIR}/bin/python" - <<'PY'
import os
import sys

import redis

try:
    redis.from_url(os.environ["REDIS_URL"], socket_timeout=5).ping()
except Exception as exc:  # noqa: BLE001 -- deploy-time probe
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
    then
        log_ok "Redis PING succeeded"
    else
        die "Redis is listening but PING failed. Check REDIS_URL credentials and the selected database index."
    fi
}

# ---------------------------------------------------------------------------
# 12. PostgreSQL
# ---------------------------------------------------------------------------
manage_postgres_decision() {
    case "${ORPHUS_MANAGE_POSTGRES:-auto}" in
        yes) printf 'yes' ;;
        no)  printf 'no'  ;;
        *)
            if is_local_host "$(pg_host)" && ! tcp_open "$(pg_host)" "$(pg_port)" 2; then
                printf 'yes'
            else
                printf 'no'
            fi
            ;;
    esac
}

find_pg_bin() {
    local d
    if have pg_config; then
        d="$(pg_config --bindir 2>/dev/null || true)"
        [[ -x "${d}/postgres" ]] && { printf '%s' "${d}"; return 0; }
    fi
    for d in $(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V -r); do
        [[ -x "${d}/postgres" ]] && { printf '%s' "${d}"; return 0; }
    done
    have postgres && { dirname "$(command -v postgres)"; return 0; }
    return 1
}

step_postgres() {
    log_step "PostgreSQL"
    local host port db user
    host="$(pg_host)"; port="$(pg_port)"; db="$(pg_database)"; user="$(pg_user)"

    if tcp_open "${host}" "${port}" 3; then
        log_ok "already listening on ${host}:${port}"
    elif [[ "$(manage_postgres_decision)" == "yes" ]]; then
        start_local_postgres "${port}"
    else
        log_error "PostgreSQL is unreachable at ${host}:${port} and this host is not configured to manage it."
        log_error "  Either start it, or set ORPHUS_MANAGE_POSTGRES=yes, or point DATABASE_URL at a managed instance."
        log_error "  A pod-local database is the wrong choice for production: it dies with the pod."
        exit 1
    fi
}

start_local_postgres() {
    local port="$1"
    local pgbin pgdata pguser
    pgbin="$(find_pg_bin)" || die "DATABASE_URL points at localhost but no postgres binary was found. Install it (${SUDO:+sudo }apt-get install -y postgresql) or point DATABASE_URL at a managed instance."
    # NOT under RUN_DIR by default: on RunPod the repo (and therefore RUN_DIR)
    # lives on a MooseFS network volume where chown fails with EPERM and initdb
    # cannot read its own cluster. PGDATA must be on the local container disk.
    pgdata="${PGDATA:-${ORPHUS_PGDATA:-/var/lib/orphus-pgdata}}"
    pguser="${ORPHUS_PG_RUN_USER:-postgres}"

    id -u "${pguser}" >/dev/null 2>&1 \
        || die "PostgreSQL refuses to run as root and the '${pguser}' account does not exist. Create it, set ORPHUS_PG_RUN_USER, or use a managed database."

    if [[ ! -s "${pgdata}/PG_VERSION" ]]; then
        log_info "initialising a new cluster at ${pgdata}"
        ensure_dir "${pgdata}"
        chown "${pguser}" "${pgdata}"
        chmod 0700 "${pgdata}"
        ${SUDO} su -s /bin/bash "${pguser}" -c \
            "'${pgbin}/initdb' -D '${pgdata}' --auth-local=trust --auth-host=scram-sha-256 -E UTF8" \
            || die "initdb failed; see the output above."
        log_ok "cluster initialised"
    else
        log_ok "cluster already initialised at ${pgdata}"
    fi

    if render_template "${ORPHUS_ROOT}/deploy/supervisor/orphus-postgres.conf.template" \
                       "${SUPERVISOR_INCLUDE_DIR}/orphus-postgres.conf" \
                       "PG_BIN=${pgbin}" \
                       "PGDATA=${pgdata}" \
                       "PG_PORT=${port}" \
                       "PG_LISTEN=${ORPHUS_PG_LISTEN:-127.0.0.1}" \
                       "PG_RUN_USER=${pguser}" \
                       "LOG_DIR=${LOG_DIR}"; then
        SUPERVISOR_DIRTY=1
        log_ok "rendered supervisor program orphus-postgres"
    else
        log_ok "supervisor program orphus-postgres already current"
    fi
    MANAGED_POSTGRES=1
}

ensure_pg_role_and_db() {
    # Only possible when we have local superuser access. On a managed database
    # the role and schema are the operator's responsibility; we just connect.
    local pguser="${ORPHUS_PG_RUN_USER:-postgres}"
    local user db pass
    user="$(pg_user)"; db="$(pg_database)"
    pass="${DATABASE_URL#*://}"; pass="${pass%%@*}"
    if [[ "${pass}" == *:* ]]; then pass="${pass#*:}"; else pass=""; fi

    have psql || return 0
    id -u "${pguser}" >/dev/null 2>&1 || return 0
    ${SUDO} su -s /bin/bash "${pguser}" -c "psql -tAc 'SELECT 1' >/dev/null 2>&1" || return 0

    # psql's :'var' interpolation quotes the value, so a password containing a
    # quote cannot break out into SQL.
    if ! ${SUDO} su -s /bin/bash "${pguser}" -c \
        "psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='${user}'\"" | grep -q 1; then
        log_info "creating role ${user}"
        # psql applies :'var' interpolation only to script/stdin input, never to
        # -c, which is sent verbatim -- the old -c form failed with
        # "syntax error at or near :". Feed the statement on stdin so the
        # interpolation (and its quoting, which is what makes a password
        # containing a quote safe) actually runs.
        ORPHUS_PG_PASS="${pass}" ${SUDO} su -s /bin/bash "${pguser}" -c \
            "psql -v ON_ERROR_STOP=1 --set=pw=\"\${ORPHUS_PG_PASS}\" <<<\"CREATE ROLE ${user} LOGIN PASSWORD :'pw';\"" \
            || die "failed to create role ${user}"
    else
        log_ok "role ${user} exists"
    fi

    if ! ${SUDO} su -s /bin/bash "${pguser}" -c \
        "psql -tAc \"SELECT 1 FROM pg_database WHERE datname='${db}'\"" | grep -q 1; then
        log_info "creating database ${db}"
        ${SUDO} su -s /bin/bash "${pguser}" -c \
            "psql -v ON_ERROR_STOP=1 -c \"CREATE DATABASE ${db} OWNER ${user};\"" \
            || die "failed to create database ${db}"
    else
        log_ok "database ${db} exists"
    fi
}

verify_postgres() {
    # asyncpg cannot parse SQLAlchemy's '+asyncpg' dialect suffix; strip it.
    local dsn="${DATABASE_URL/postgresql+asyncpg:\/\//postgresql://}"
    if ORPHUS_PG_DSN="${dsn}" "${VENV_DIR}/bin/python" - <<'PY'
import asyncio
import os
import sys

import asyncpg


async def main() -> None:
    conn = await asyncio.wait_for(asyncpg.connect(os.environ["ORPHUS_PG_DSN"]), timeout=10)
    try:
        print(await conn.fetchval("SELECT version()"))
    finally:
        await conn.close()


try:
    asyncio.run(main())
except Exception as exc:  # noqa: BLE001 -- deploy-time probe
    print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
PY
    then
        log_ok "PostgreSQL connection verified"
    else
        log_error "Could not connect with DATABASE_URL."
        log_error "  Check the role, password, database name, and pg_hba.conf host rules."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# 13. Migrations
# ---------------------------------------------------------------------------
step_migrations() {
    log_step "Database migrations"
    if [[ "${SKIP_MIGRATIONS}" -eq 1 ]]; then
        log_info "skipped (--skip-migrations)"
        return 0
    fi
    if [[ ! -f "${ORPHUS_ROOT}/alembic.ini" ]]; then
        log_warn "alembic.ini not found; skipping migrations."
        log_warn "The persistence layer has not landed yet. Once it does, this step runs"
        log_warn "'alembic upgrade head' automatically -- no change to deploy.sh needed."
        return 0
    fi
    log_info "alembic upgrade head"
    (cd "${ORPHUS_ROOT}" && "${VENV_DIR}/bin/alembic" upgrade head) \
        || die "migrations failed. Inspect with: ${VENV_DIR}/bin/alembic current && ${VENV_DIR}/bin/alembic history"
    log_ok "schema at head"
}

# ---------------------------------------------------------------------------
# 14. Supervisor
# ---------------------------------------------------------------------------
step_supervisor_config() {
    log_step "Supervisor configuration"

    # Contract: the ASGI application factory. Until the API package lands this
    # is the agreed import path -- `orphus.api.app:create_app()` returning a
    # FastAPI/ASGI app. Override wholesale with ORPHUS_API_COMMAND if the
    # entrypoint differs (e.g. once `orphus serve` exists as a console script).
    local api_cmd
    api_cmd="${ORPHUS_API_COMMAND:-${VENV_DIR}/bin/python -m uvicorn --factory orphus.api.app:create_app --host ${HOST} --port ${PORT} --workers ${ORPHUS_API_WORKERS:-1} --timeout-graceful-shutdown ${ORPHUS_SHUTDOWN_GRACE_S:-20}}"

    if render_template "${ORPHUS_ROOT}/deploy/supervisord.conf.template" \
                       "${SUPERVISOR_CONF}" \
                       "ORPHUS_ROOT=${ORPHUS_ROOT}" \
                       "LOG_DIR=${LOG_DIR}" \
                       "SUPERVISOR_SOCK=${SUPERVISOR_SOCK}" \
                       "SUPERVISOR_PID=${SUPERVISOR_PID}" \
                       "SUPERVISOR_INCLUDE_DIR=${SUPERVISOR_INCLUDE_DIR}"; then
        SUPERVISOR_DIRTY=1
        SUPERVISOR_MASTER_CHANGED=1
        log_ok "rendered ${SUPERVISOR_CONF}"
    else
        log_ok "supervisord.conf already current"
    fi

    if render_template "${ORPHUS_ROOT}/deploy/supervisor/orphus-api.conf.template" \
                       "${SUPERVISOR_INCLUDE_DIR}/orphus-api.conf" \
                       "API_COMMAND=${api_cmd}" \
                       "ORPHUS_ROOT=${ORPHUS_ROOT}" \
                       "LOG_DIR=${LOG_DIR}" \
                       "RUN_USER=${RUN_USER}" \
                       "VENV_DIR=${VENV_DIR}" \
                       "API_STARTSECS=${ORPHUS_API_STARTSECS:-30}" \
                       "API_STOPWAIT=${ORPHUS_API_STOPWAIT:-45}"; then
        SUPERVISOR_DIRTY=1
        log_ok "rendered supervisor program orphus-api"
    else
        log_ok "supervisor program orphus-api already current"
    fi
}

step_permissions() {
    log_step "Script permissions"
    # Fresh clones on filesystems without an exec bit (or via a zip export) land
    # non-executable. Fix it here so the operator scripts always run.
    local f n=0
    chmod +x "${ORPHUS_ROOT}/deploy.sh" 2>/dev/null || true
    for f in "${ORPHUS_ROOT}"/scripts/*.sh; do
        [[ -f "${f}" ]] || continue
        [[ "$(basename "${f}")" == _* ]] && continue   # sourced helpers stay non-exec
        if [[ ! -x "${f}" ]]; then chmod +x "${f}"; n=$((n + 1)); fi
    done
    log_ok "operator scripts executable (${n} updated)"
}

step_launch() {
    log_step "Launching services"
    if [[ "${NO_START}" -eq 1 ]]; then
        log_info "skipped (--no-start). Launch later with: ./scripts/start.sh"
        return 0
    fi

    if supervisor_running; then
        if [[ "${SUPERVISOR_MASTER_CHANGED:-0}" -eq 1 ]]; then
            # A changed master config needs a full reload; `update` only picks
            # up program-level changes.
            log_info "supervisord config changed; reloading"
            sctl reload || true
            sleep 3
        elif [[ "${SUPERVISOR_DIRTY:-0}" -eq 1 ]]; then
            log_info "program definitions changed; applying"
            sctl update || true
        else
            log_ok "supervisord already running with the current configuration"
        fi
    else
        log_info "starting supervisord"
        "$(supervisord_bin)" -c "${SUPERVISOR_CONF}" \
            || die "supervisord failed to start. Check ${LOG_DIR}/supervisord.log"
        sleep 2
    fi

    # `start all` returns non-zero when a program is already running; that is
    # the idempotent case, not a failure.
    sctl start all >/dev/null 2>&1 || true
    sctl status || true
}

# ---------------------------------------------------------------------------
# 15. Health
# ---------------------------------------------------------------------------
step_health() {
    log_step "Health checks"
    if [[ "${NO_START}" -eq 1 ]]; then log_info "skipped (--no-start)"; return 0; fi

    local attempts=$(( HEALTH_TIMEOUT_S / 5 ))
    [[ "${attempts}" -ge 1 ]] || attempts=1

    log_info "waiting up to ${HEALTH_TIMEOUT_S}s for ${BASE_URL}/health"
    log_info "(a cold vLLM load of the 3B TTS model dominates this wait)"

    local i status
    for ((i = 1; i <= attempts; i++)); do
        status="$(http_status "${BASE_URL}/health" 5)"
        case "${status}" in
            2*)
                log_ok "GET /health -> ${status}"
                probe_readiness
                probe_metrics
                return 0
                ;;
            404)
                log_warn "GET /health -> 404. The process is serving but has no /health route."
                log_warn "Contract: the API layer must expose GET /health (liveness) and"
                log_warn "GET /health/ready (readiness). See docs/api-reference.md."
                return 0
                ;;
            000)
                : # not listening yet
                ;;
            *)
                log_debug "GET /health -> ${status}"
                ;;
        esac
        # Fail fast if supervisor has already given up, rather than burning the
        # whole timeout on a process that will never come back.
        if sctl status "${SVC_API}" 2>/dev/null | grep -qE 'FATAL|EXITED'; then
            log_error "${SVC_API} is not running:"
            sctl status "${SVC_API}" >&2 || true
            log_error "  Last 40 lines of ${LOG_DIR}/api.err.log:"
            tail -40 "${LOG_DIR}/api.err.log" 2>/dev/null | sed 's/^/         /' >&2 || true
            log_error "  Runbook: docs/runbook.md -> 'API worker crash-loop'"
            exit 1
        fi
        sleep 5
    done

    log_error "API did not become healthy within ${HEALTH_TIMEOUT_S}s (last status ${status})."
    log_error "  ./scripts/logs.sh api --tail 100"
    log_error "  ./scripts/status.sh"
    log_error "  Raise the budget: ./deploy.sh --skip-deps --skip-models --health-timeout 900"
    exit 1
}

probe_readiness() {
    local status
    status="$(http_status "${BASE_URL}/health/ready" 10)"
    case "${status}" in
        2*)   log_ok   "GET /health/ready -> ${status} (models loaded)" ;;
        404)  log_warn "GET /health/ready -> 404 (readiness route not implemented yet)" ;;
        503)  log_warn "GET /health/ready -> 503: live but not ready. Models may still be loading." ;;
        *)    log_warn "GET /health/ready -> ${status}" ;;
    esac
}

probe_metrics() {
    local path="${ORPHUS_METRICS_PATH:-/metrics}" status
    status="$(http_status "${BASE_URL}${path}" 5)"
    if [[ "${status}" =~ ^2 ]]; then
        log_ok "GET ${path} -> ${status} (Prometheus scrape target live)"
    else
        log_warn "GET ${path} -> ${status}; Prometheus will have nothing to scrape."
    fi
}

# ---------------------------------------------------------------------------
# 16. Summary
# ---------------------------------------------------------------------------
step_summary() {
    local elapsed=$(( $(date +%s) - DEPLOY_STARTED_AT ))
    printf '\n%s%s%s\n' "${C_GREEN}${C_BOLD}" "==========================================================" "${C_RESET}"
    printf '%s Orphus deployed in %dm %02ds%s\n' "${C_GREEN}${C_BOLD}" $((elapsed / 60)) $((elapsed % 60)) "${C_RESET}"
    printf '%s%s%s\n\n' "${C_GREEN}${C_BOLD}" "==========================================================" "${C_RESET}"

    local ext="${ORPHUS_PUBLIC_URL:-}"
    if [[ -z "${ext}" && -n "${RUNPOD_POD_ID:-}" ]]; then
        # RunPod's HTTP proxy hostname is derived from the pod id and port.
        ext="https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net"
    fi

    printf '  %sAPI%s          %s\n'  "${C_BOLD}" "${C_RESET}" "${BASE_URL}"
    printf '  %sHealth%s       %s\n'  "${C_BOLD}" "${C_RESET}" "${BASE_URL}/health"
    printf '  %sReadiness%s    %s\n'  "${C_BOLD}" "${C_RESET}" "${BASE_URL}/health/ready"
    printf '  %sMetrics%s      %s\n'  "${C_BOLD}" "${C_RESET}" "${BASE_URL}${ORPHUS_METRICS_PATH:-/metrics}"
    printf '  %sOpenAPI%s      %s\n'  "${C_BOLD}" "${C_RESET}" "${BASE_URL}/docs"
    printf '  %sVoice WS%s     %s\n'  "${C_BOLD}" "${C_RESET}" "${BASE_URL/http/ws}/v1/stream"
    [[ -n "${ext}" ]] && printf '  %sPublic%s       %s\n' "${C_BOLD}" "${C_RESET}" "${ext}"

    printf '\n  %sLogs%s         %s\n' "${C_BOLD}" "${C_RESET}" "${LOG_DIR}"
    printf '  %sModels%s       %s\n'   "${C_BOLD}" "${C_RESET}" "${MODEL_ROOT}"
    printf '  %sSupervisor%s   %s\n'   "${C_BOLD}" "${C_RESET}" "${SUPERVISOR_CONF}"
    printf '  %sEnvironment%s  %s\n'   "${C_BOLD}" "${C_RESET}" "${ORPHUS_ENV}"

    printf '\n  Next:\n'
    printf '    ./scripts/status.sh              service and GPU status\n'
    printf '    ./scripts/health.sh              detailed component health\n'
    printf '    ./scripts/logs.sh api -f         follow the API log\n'
    printf '    ./scripts/benchmark.sh --help    latency and load harness\n\n'
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
main() {
    printf '%s%sOrphus deploy%s  (env=%s, root=%s)\n' \
        "${C_BOLD}" "${C_CYAN}" "${C_RESET}" "${ORPHUS_ENV}" "${ORPHUS_ROOT}"

    step_directories        # 0. dirs first: everything below writes into them
    step_gpu                # 1.
    step_cuda               # 2.
    step_python             # 3.
    step_os_packages        # 4.
    step_venv               # 5.
    step_dependencies       # 6.
    step_patch_orpheus      #    upstream wheel predates the API we call
    step_verify_torch       # 7.
    step_models             # 8.
    step_vendor             #    optional reference clones
    step_config             # 9.
    validate_environment    # 10.
    step_redis              # 11.
    step_postgres           # 12.
    step_permissions
    step_supervisor_config

    # Ordering note: when this host manages Redis/PostgreSQL itself, they are
    # supervisor programs -- so supervisor has to come up before the datastores
    # can be verified and before migrations can run. Externally managed
    # datastores are already up and are verified before anything is launched.
    if [[ "${NO_START}" -eq 1 ]]; then
        if [[ "${MANAGED_REDIS:-0}" -eq 0 && "${MANAGED_POSTGRES:-0}" -eq 0 ]]; then
            step_datastores_and_migrations
        else
            log_warn "--no-start with pod-local datastores: skipping verification and migrations."
            log_warn "Run ./scripts/start.sh, then ./deploy.sh --skip-deps --skip-models to finish."
        fi
        step_launch
        step_summary
        return 0
    fi

    step_launch                     # 14. supervisor brings up the process group
    if [[ "${MANAGED_REDIS:-0}" -eq 1 ]]; then
        wait_for_tcp_or_die "$(redis_host)" "$(redis_port)" "Redis"
    fi
    if [[ "${MANAGED_POSTGRES:-0}" -eq 1 ]]; then
        wait_for_tcp_or_die "$(pg_host)" "$(pg_port)" "PostgreSQL"
    fi
    step_datastores_and_migrations  # 13.
    step_health                     # 15.
    step_summary                    # 16.
}

step_datastores_and_migrations() {
    verify_redis
    ensure_pg_role_and_db
    verify_postgres
    step_migrations
}

wait_for_tcp_or_die() {
    local host="$1" port="$2" label="$3" i
    for ((i = 1; i <= 30; i++)); do
        tcp_open "${host}" "${port}" 2 && { log_ok "${label} accepting connections on ${host}:${port}"; return 0; }
        sleep 2
    done
    log_error "${label} did not start within 60s."
    log_error "  ./scripts/logs.sh ${label,,} --tail 50"
    exit 1
}

main "$@"

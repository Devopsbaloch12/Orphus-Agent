# Orphus Voice AI

Orphus is a single-GPU realtime voice platform using Silero VAD, NVIDIA
Nemotron 3.5 ASR, Grok, and Orpheus TTS. The production process keeps model
weights resident and creates isolated VAD/ASR/pipeline state for every call.

## RunPod deployment

Requirements: Ubuntu 22.04+, Python 3.12, an NVIDIA GPU with 48 GB VRAM, CUDA,
and outbound access to the model registries and xAI.

```bash
git clone <repository-url> orphus
cd orphus
cp .env.example .env
# Put GROK_API_KEY and ORPHUS_API_KEYS in the environment or .env.secrets.
chmod +x deploy.sh scripts/*.sh
./deploy.sh
```

Verify with `./scripts/status.sh` and `./scripts/health.sh`. The API listens on
port 8000 by default. Create a session with `POST /v1/sessions`, then connect
16 kHz mono PCM16 audio to `/v1/ws/{session_id}`. Returned binary frames are
24 kHz mono PCM16.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src tests
mypy src
```

GPU packages and weights are deliberately lazy-imported, so CPU-only unit
tests do not need CUDA. See [the operations runbook](docs/runbook.md) before
running production traffic.


# Orphus Voice API — client & load test

Base URL: $ORPHUS_URL
Auth:     header `x-api-key: $ORPHUS_KEY`
          (required on REST calls AND on the WebSocket handshake)

## Call flow (per call)
1. POST /v1/sessions        {"voice":"tara"}  -> 201 {"session_id":"..."}
                                              -> 503 when 20 sessions are already active
2. WS   /v1/ws/{session_id} with the x-api-key header
3. SEND raw binary frames: 16 kHz mono PCM16 LE, 20 ms = 640 bytes, paced in realtime
4. RECV raw binary frames: 24 kHz mono PCM16 LE  -> play to caller
5. Close WS to hang up   (optional: DELETE /v1/sessions/{id})

GET /health  -> no auth, use for monitoring

Voices: tara leah jess mia zoe (female) | leo dan zac (male)

## Run the 20-concurrent test
    pip install websockets aiohttp
    ./loadtest.sh caller.wav 20

Prints, per call: reply audio length + response latency
(end-of-caller-speech -> first agent audio byte). That latency is the
number to compare against your 2 s target.

## ViciDial / Asterisk
Asterisk SLIN is 8 kHz. Upsample x2 toward Orphus, downsample the 24 kHz
replies x3 toward the channel, or audio will be pitch-shifted.


## Configure

    export ORPHUS_URL=https://<your-pod>-8000.proxy.runpod.net
    export ORPHUS_KEY=<your api key>
    pip install websockets aiohttp
    python orphus_client.py --wav caller.wav --out reply.wav
    python orphus_client.py --wav caller.wav --concurrency 20

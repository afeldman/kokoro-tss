# kokoro-tts: Local Text-to-Speech Service

**Status:** v0.1.0  
**Platform:** macOS Apple Silicon (M4 Max) with MPS support  
**License:** MIT (Kokoro model subject to its own license)

[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![uv](https://img.shields.io/badge/build-uv-brightgreen)](https://github.com/astral-sh/uv)

---

## Overview

`kokoro-tts` is a local Text-to-Speech service that uses the [Kokoro](https://github.com/hexgrad/Kokoro) model for neural audio synthesis. It provides:

- **FastAPI HTTP API** for programmatic access
- **Typer CLI** for command-line usage
- **Local inference** with no external API calls
- **WAV output** (PCM 16-bit, 24000 Hz, mono)
- **Apple MPS acceleration** on Mac Silicon

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  kokoro-tts Service                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐     ┌──────────┐│
│  │   CLI (Typer) │    │  HTTP API    │     │ Fake     ││
│  │  kokoro-tts  │────▶│ (FastAPI)    │◀───▶│ Backend  ││
│  └──────────────┘     └──────────────┘     │(for tests)││
│                         │                  └──────────┘│
│                   ┌─────▼────────────────────────┐      │
│                   │   SpeechService              │      │
│                   │   (shared business logic)    │      │
│                   └─────────────────────────────┘      │
│                         │                               │
│                   ┌─────▼────────────────────────┐      │
│                   │  InferenceBackend            │      │
│                   │  (Kokoro KPipeline adapter)  │      │
│                   └─────────────────────────────┘      │
│                         │                               │
│                   ┌─────▼────────────────────────┐      │
│                   │      Kokoro KPipeline        │      │
│                   │  (hexgrad/Kokoro-82M)        │      │
│                   └─────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Module | Responsibility |
|--------|----------------|
| `config.py` | Pydantic Settings, environment variables, device detection |
| `schemas.py` | Request/Response Pydantic models |
| `inference.py` | Kokoro KPipeline adapter with thread-safe inference |
| `service.py` | Shared business logic for API and CLI |
| `audio.py` | WAV encoding (16-bit PCM) |
| `api.py` | FastAPI routes and error handling |
| `cli.py` | Typer command-line interface |
| `exceptions.py` | Custom exception hierarchy |

---

## System Requirements

### Hardware

- **CPU:** Apple Silicon (M1/M2/M4) with MPS support, or Intel CPU
- **RAM:** Minimum 8 GB (recommended 16 GB+)
- **Disk:** ~315 MB for the model weights and the preloaded voice (measured: 313 MB for one revision), held in the shared Hugging Face cache

### Software

- **macOS** 12.0 (Monterey) or later
- **Python** 3.13+ (via uv)
- **PyTorch** with MPS support (Apple Silicon)
- **Phonemizer library** — the native `espeak-ng` library and its data ship inside the `espeakng-loader` wheel that comes with `misaki[en]`, so no separate Homebrew installation is required. (Verified on this machine: `espeak-ng` is not in `PATH` and synthesis works.) A system install is a fallback only.

---

## Installation

### Prerequisites

```bash
# Install uv (project manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Project Setup

```bash
cd /Users/anton.feldmann/backup/Projects/comp/kokoro-tts

# Sync dependencies with uv
uv sync

# Check Python version
cat .python-version  # Should show 3.13
```

### Phonemizer library (usually nothing to do)

The phonemizer's native library does not have to be installed system-wide: it is
bundled with the Python package `espeakng-loader`, which `uv sync` installs.

```bash
# The bundled library and its data live inside the project virtualenv
ls .venv/lib/python*/site-packages/espeakng_loader/libespeak-ng.dylib
```

Only if that bundled library cannot be loaded (or if the phonemizer warns about
skipped out-of-dictionary words) install a system copy as a fallback:

```bash
brew install espeak-ng
```

---

## Configuration

### Environment Variables

Create a `.env` file (not tracked in git):

```bash
# Server configuration
kokoro_HOST=127.0.0.1
kokoro_PORT=8765
kokoro_LOG_LEVEL=INFO

# Model configuration  
kokoro_MODEL_ID=hexgrad/Kokoro-82M
kokoro_MODEL_DEVICE=auto  # auto, mps, or cpu
kokoro_MODEL_CACHE_DIR=~/.cache/kokoro

# Voice and language defaults
kokoro_DEFAULT_VOICE=af_heart
kokoro_DEFAULT_LANGUAGE=en-us

# Validation limits
kokoro_MAX_TEXT_LENGTH=5000

# Request handling
kokoro_REQUEST_TIMEOUT_SECONDS=60
```

### Device Selection

- **`auto`** (default): Automatically detects MPS on Apple Silicon, falls back to CPU
- **`mps`**: Force use of Metal Performance Shaders (Apple GPU)
- **`cpu`**: Use CPU (slower but compatible with all devices)

The device is selected through the `kokoro_MODEL_DEVICE` environment variable only; there is no `--device` CLI flag. An explicitly set `mps` or `cpu` is never silently overridden by auto-detection.

---

## Usage

### Start the HTTP Server

```bash
uv run kokoro-tts serve
```

Options:

- `--host, -h`: Host to bind to (default: 127.0.0.1)
- `--port, -p`: Port number (default: 8765)
- `--log-level, -l`: Logging level (debug, info, warning, error)

The server loads the model on startup: about 4 s from the cache on this machine (measured). The very first run additionally downloads the weights, so it takes as long as that download needs.

### Generate Speech via CLI

`TEXT` is a **positional** argument (there is no `--text` flag). `--language/-l`, `--voice/-v` and `--output/-o` are optional.

```bash
# Basic usage: text first, then options
uv run kokoro-tts generate "Hello from local Kokoro!" \
  --language en-us \
  --voice af_heart \
  --output output.wav
```

Output: Valid WAV file at `output.wav` (`--output/-o` defaults to `output.wav`).

### List Available Voices

```bash
uv run kokoro-tts voices

# Filter by language
uv run kokoro-tts voices --language en-us
```

This lists the voices loaded in the current process. `af_heart` is cached out of the box; every other voice is downloaded from Hugging Face on first use and then served from the cache.

---

## Daily workflow

The day-to-day loop for a working installation. Every command runs through `uv run`, so the project's virtual environment is used consistently.

### 1. Set up the project environment

```bash
# From the project root
cd /Users/anton.feldmann/backup/Projects/comp/kokoro-tts

# Create/refresh .venv and install the pinned dependencies
uv sync
```

`uv sync` builds `.venv/` from the lockfile and brings everything the phonemizer needs: the native `espeak-ng` library ships in the `espeakng-loader` wheel, so no separate system package is required (see [Phonemizer library](#phonemizer-library-usually-nothing-to-do)).

### 2. Start the server

```bash
uv run kokoro-tts serve
```

```
Starting Kokoro TTS server...
  Host: 127.0.0.1
  Port: 8765
  Language: en-us
  Voice: af_heart
  Device: mps

Loading Kokoro model...
Model loaded successfully!

Starting server at http://127.0.0.1:8765
Press Ctrl+C to stop
```

The model is loaded exactly once here, at startup; requests never re-load or re-download it. `--host/-h`, `--port/-p` and `--log-level/-l` override the corresponding `kokoro_*` settings for that run, otherwise the settings defaults apply. The server binds `127.0.0.1` by default and is therefore reachable only from this machine.

Leave this shell running and use a second terminal for the steps below.

### 3. Check health and voices

Health answers HTTP 200 whenever the process is up, even before the model is ready:

```bash
curl http://127.0.0.1:8765/health
# identical handler on the versioned path:
curl http://127.0.0.1:8765/v1/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps",
  "backend": "kokoro",
  "language": "en-us"
}
```

`status`/`model_loaded` separate "process running" from "model ready to serve". Voices are listed only once the model is loaded (HTTP 503 before that):

```bash
curl http://127.0.0.1:8765/v1/audio/voices
```

```json
{
  "voices": [
    {"id": "af_heart", "language": "en-us"}
  ]
}
```

This reports the voices actually loaded in the process. `af_heart` is cached out of the box; any other voice is fetched from Hugging Face on first use and served from the cache afterwards.

### 4. Synthesize speech

Via the CLI (a separate process that loads the model itself):

```bash
uv run kokoro-tts generate "Hello from local Kokoro!" --output hello.wav
```

`TEXT` is positional; there is no `--text` flag. `--language/-l`, `--voice/-v` and `--output/-o` are optional and fall back to the configured language/voice and `output.wav`.

Via HTTP (uses the already-running server and its loaded model):

```bash
curl -X POST http://127.0.0.1:8765/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from local Kokoro!", "voice": "af_heart"}' \
  --output hello_http.wav
```

The endpoint returns **binary WAV**, not JSON: `Content-Type: audio/wav` and `Content-Disposition: attachment; filename="speech.wav"`. Write it straight to a file with `--output` (or the equivalent in your client).

**Performance note (measured on MPS):** generating a short sentence takes roughly 0.1–0.35 s per request; long texts scale with their length (585 characters yielded ~36.3 s of audio and took correspondingly longer). The server runs inference off the event loop in a bounded thread pool (40 worker threads) while the backend holds an inference lock, so concurrent requests are queued and served one at a time rather than running fully in parallel.

### 5. Stop the server

Press `Ctrl+C` in the shell running `serve`. The process exits cleanly; the next start re-loads the model from cache.

### 6. Common problems and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ERROR: Failed to load Kokoro model: ...` at startup | No network on the very first run | Connect once so the weights download, then the server runs offline |
| Startup fails with an eSpeak / phonemizer error | bundled `espeakng-loader` library cannot be loaded | install a system copy as a fallback: `brew install espeak-ng` |
| `status: "unhealthy"`, voices return 503 | Model still loading | Wait for `Model loaded successfully!`, then retry |
| 400 `Language is not available on this instance` | Requested a language other than the one the server was started with | Start the server with `kokoro_DEFAULT_LANGUAGE=<lang>` (or use `generate --language <lang>`, which builds a matching instance) |
| 404 `Voice is not available` | Unknown voice id, or a not-yet-downloaded voice while offline | Check the id; use `af_heart`, or go online once to fetch the voice |
| 422 on `POST /v1/audio/speech` | Body missing `input`, or text over `kokoro_MAX_TEXT_LENGTH` | Send a valid `input` field (1..`kokoro_MAX_TEXT_LENGTH` chars) |
| Device shows `cpu` on Apple Silicon | MPS unavailable | Use a PyTorch build with MPS support, or accept CPU via `kokoro_MODEL_DEVICE=cpu` |
| Port already in use | Another instance is running | Stop it, or start with `--port 8766` |

There is no `--device` CLI flag; select the device with the `kokoro_MODEL_DEVICE` environment variable.

---

## HTTP API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check (process up) |
| GET | `/v1/health` | Same handler on the versioned path |
| GET | `/v1/audio/voices` | List the voices loaded in this process (503 until the model is loaded) |
| POST | `/v1/audio/speech` | Synthesize speech from text; returns **binary WAV** |

### Health Check

```bash
curl http://localhost:8765/health
# identical handler:
curl http://localhost:8765/v1/health
```

Answers HTTP 200 whenever the process is up, even before the model is ready. `status`/`model_loaded` report whether the model is ready to serve:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps",
  "backend": "kokoro",
  "language": "en-us"
}
```

### List Voices

```bash
curl http://localhost:8765/v1/audio/voices
```

Returns HTTP 503 until the model is loaded. Once loaded it reports only the voices actually loaded in the process (with their language), so any id shown here can be synthesized without a download. `af_heart` is cached out of the box; every other voice is fetched on first use.

```json
{
  "voices": [
    {"id": "af_heart", "language": "en-us"}
  ]
}
```

### Generate Speech

```bash
curl -X POST http://localhost:8765/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test of Kokoro TTS.",
    "language": "en-us",
    "voice": "af_heart",
    "response_format": "wav"
  }' \
  --output output.wav
```

The response body is **binary WAV**, not JSON: `Content-Type: audio/wav` and `Content-Disposition: attachment; filename="speech.wav"`. Only error responses are JSON.

---

## Request/Response Schemas

### SpeechRequest

```json
{
  "input": string,           // Required. Text to synthesize (1..kokoro_MAX_TEXT_LENGTH)
  "language": string|null,   // Optional. Default: kokoro_DEFAULT_LANGUAGE ("en-us")
  "voice": string|null,      // Optional. Default: kokoro_DEFAULT_VOICE ("af_heart")
  "response_format": string  // Optional. Only "wav" is supported (default: "wav")
}
```

### HealthResponse

```json
{
  "status": string,          // "healthy" or "unhealthy"
  "model_loaded": boolean,   // Whether model is initialized
  "device": string,          // "mps", "cpu", or "unknown"
  "backend": string,         // "kokoro"
  "language": string|null    // Language this instance was started with
}
```

### VoicesResponse

```json
{
  "voices": [
    {
      "id": string,          // Voice identifier (e.g., "af_heart")
      "language": string|null // Language of the voice (e.g., "en-us")
    }
  ]
}
```

---

## Error Handling

| HTTP Code | Exception | Description |
|-----------|-----------|-------------|
| 400 | TextValidationError | Empty/whitespace-only or too-long text (> `kokoro_MAX_TEXT_LENGTH`) |
| 400 | LanguageNotFoundError | Language not available on this instance |
| 404 | VoiceNotFoundError | Unknown voice |
| 415 | UnsupportedFormatError | `response_format` other than `"wav"` |
| 422 | Pydantic ValidationError | Invalid request schema (e.g. missing `input`, or text over `kokoro_MAX_TEXT_LENGTH`) |
| 503 | ModelNotLoadedError | Model not loaded yet |
| 500 | InferenceError | Backend inference failure (generic message; the cause stays in the server log) |

### Example Errors

```json
// Empty text (400 Bad Request)
{
  "detail": "Input text cannot be empty"
}

// Voice not found (404 Not Found)
{
  "detail": "Voice is not available: unknown_voice"
}

// Unavailable language (400 Bad Request)
{
  "detail": "Language is not available on this instance: ja (instance language: en-us)"
}

// Model not loaded (503 Service Unavailable)
{
  "detail": "Model is not loaded yet. Please restart the server."
}
```

---

## Model Information

### Kokoro Model

- **Repository:** hexgrad/Kokoro-82M on Hugging Face
- **Architecture:** Transformer-based TTS model (82M parameters)
- **License:** See [Kokoro LICENSE](https://github.com/hexgrad/Kokoro/blob/main/LICENSE)
- **Usable languages:** en-us, en-gb, es, fr-fr, hi, it, pt-br
- **Not usable here:** ja (Japanese) and zh (Mandarin) fail to initialize, because their optional G2P dependencies (`pyopenjtalk`, `ordered_set`) are not installed

### Model Download and Cache

- The weights come from `hexgrad/Kokoro-82M` through the **Hugging Face cache**: the model folder `~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M` holds only symlinks, the actual bytes live as shared blobs under `~/.cache/huggingface/hub/blobs/`. Measured size for the revision in use: 313 MB (`kokoro-v1_0.pth` 312 MB, `voices/af_heart.pt` 0.5 MB).
- The first run downloads them; later runs read them from the cache. The cache is shared with other Hugging Face models, so `du` on the hub directory reports more than Kokoro alone.
- The model is loaded exactly once per server start, never per request.
- `kokoro_MODEL_CACHE_DIR` (`~/.cache/kokoro`) is a projected working directory: it is created at startup but holds no weights of its own.
- **Offline behaviour:** after one successful download the server runs without a network connection (the model and already-loaded voices come from the cache). A voice that has never been loaded needs the network and fails with HTTP 404 otherwise.

---

## Audio Format

| Attribute | Value | Source |
|-----------|-------|--------|
| Sample Rate | 24000 Hz | Kokoro-82M vocoder |
| Channels | 1 (mono) | Model output |
| Bit Depth | 16-bit PCM | WAV encoding |
| Format | RIFF/WAVE/PCM | Standard WAV |

**Verified output:** a request to `POST /v1/audio/speech` produces a file that `file` reports as `RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 24000 Hz` (measured on this machine: 129,644 bytes / 2.700 s for the sample sentence used in the daily workflow).

The rate is a property of the model, not a header we may choose: Kokoro advances 600 audio samples per `pred_dur` frame and its pipeline documents 40 frames per second (`kokoro/pipeline.py`), i.e. 24000 Hz. Measuring this instead of trusting the header is what fixed an earlier version that labelled the audio 22050 Hz and therefore played it back ~8.8 % too slow and ~1.5 semitones too low.

---

## Testing

Run the test suite without model weights (uses the FakeBackend; 91 tests, no download):

```bash
uv run pytest tests/ -v
```

Lint and format checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

Test coverage includes:
- Health endpoint
- Voice listing
- Speech synthesis
- WAV header validation
- Error responses
- CLI commands

---

## Troubleshooting

### Model Not Loading

**Symptoms:** Server fails to start after "Loading Kokoro model..."

**Solutions:**
```bash
# The phonemizer library ships with the project (espeakng-loader) - no system package needed
ls .venv/lib/python*/site-packages/espeakng_loader/libespeak-ng.dylib

# Check the internet connection (only needed for the first download)
ping -c1 huggingface.co

# Re-download the weights: they live in the Hugging Face cache, not in ~/.cache/kokoro
rm -rf ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M
uv run kokoro-tts serve
```

### MPS Not Available

**Symptoms:** Device shows "cpu" instead of "mps"

**Solutions:**
```bash
# Check PyTorch MPS support
python3 -c "import torch; print(torch.backends.mps.is_available())"

# Force CPU if MPS is not supported (environment variable; there is no --device flag)
export kokoro_MODEL_DEVICE=cpu
uv run kokoro-tts serve
```

### Voice Not Found

**Symptoms:** HTTP 404 with `"detail": "Voice is not available: <voice_id>"`

**Solutions:**
- Voices other than `af_heart` are downloaded on first use and need the network for that first fetch (offline: HTTP 404)
- Check the id and the currently loaded voices: `uv run kokoro-tts voices`
- Ensure the model is fully loaded (wait for "Model loaded successfully!")

---

## Security Notes

- **Default bind:** `127.0.0.1` (localhost only)
- **0.0.0.0** binding requires explicit `--host 0.0.0.0` flag
- No external API calls; all processing is local
- Model weights downloaded from Hugging Face Hub

---

## License

This project uses the MIT license. The Kokoro model is subject to its own license ([hexgrad/Kokoro LICENSE](https://github.com/hexgrad/Kokoro/blob/main/LICENSE)).

By using this service, you agree to comply with the model's license terms and any usage restrictions associated with the audio output.

---

## Contributing

This is an internal project for scientific research purposes. Feature requests should be directed to the maintainer.

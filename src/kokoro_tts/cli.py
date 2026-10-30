"""
Typer CLI commands for kokoro-tts.

``serve`` starts the HTTP server through :func:`kokoro_tts.app.create_app`, so
the running server serves exactly the router the test suite exercises.
``generate`` and ``serve`` share :class:`kokoro_tts.service.SpeechService`.
"""

import logging
import sys
from pathlib import Path

import typer

from kokoro_tts.config import Settings, get_settings, parse_device
from kokoro_tts.exceptions import KokoroTTSException
from kokoro_tts.inference import InferenceBackend
from kokoro_tts.service import SpeechService

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="kokoro-tts",
    help="Local Kokoro Text-to-Speech Service",
    add_completion=False,
)


def build_backend(settings: Settings, lang_code: str | None = None) -> InferenceBackend:
    """
    Build an inference backend from settings.

    This only constructs the adapter; ``InferenceBackend.load()`` performs the
    actual (single) model initialization.
    """
    return InferenceBackend(
        lang_code=lang_code or settings.DEFAULT_LANGUAGE,
        repo_id=settings.MODEL_ID,
        device=parse_device(settings.MODEL_DEVICE),
        cache_dir=Path(settings.MODEL_CACHE_DIR).expanduser(),
        default_voice=settings.DEFAULT_VOICE,
    )


def get_backend(settings: Settings | None = None) -> InferenceBackend:
    """Backend from the effective settings (kept for backward compatibility)."""
    return build_backend(settings or get_settings())


def _load_backend(backend: InferenceBackend, advice: bool = True) -> bool:
    """Load the model once; report a failed start instead of raising."""
    try:
        print("\nLoading Kokoro model...")
        backend.load()
        print("Model loaded successfully!")
        return True
    except Exception as exc:  # noqa: BLE001 - startup boundary: report and exit
        logger.error("Failed to load model: %s", exc)
        print(f"\nERROR: Failed to load Kokoro model: {exc}")
        if advice:
            print("Please ensure:")
            print("  - Internet connection is available for the first download")
            print("  - Disk space is sufficient (~350MB for model and voices)")
            print("  - espeak-ng is installed (brew install espeak-ng)")
        return False


@app.command("serve")
def serve(
    host: str | None = typer.Option(
        None,
        "--host",
        "-h",
        help="Host to bind to (default: kokoro_HOST, 127.0.0.1)",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        "-p",
        help="Port to bind to (default: kokoro_PORT, 8765)",
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Logging level (default: kokoro_LOG_LEVEL, info)",
    ),
):
    """Start the FastAPI HTTP server (loads the model once at startup)."""
    # Resolved at call time, so the environment of the process that actually
    # starts the server applies (an import-time literal would freeze it).
    settings = get_settings()
    host = host or settings.HOST
    port = port or settings.PORT
    log_level = (log_level or settings.LOG_LEVEL).lower()

    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    device = parse_device(settings.MODEL_DEVICE)

    print("Starting Kokoro TTS server...")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Language: {settings.DEFAULT_LANGUAGE}")
    print(f"  Voice: {settings.DEFAULT_VOICE}")
    print(f"  Device: {device}")

    # Model initialization happens here, exactly once per process.
    backend = build_backend(settings)
    if not _load_backend(backend):
        sys.exit(1)

    # The HTTP layer is the tested router, mounted by the application factory.
    from kokoro_tts.app import create_app

    api = create_app(backend, settings)

    import uvicorn

    print(f"\nStarting server at http://{host}:{port}")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        api,
        host=host,
        port=port,
        log_level=log_level.lower(),
        reload=False,
    )


@app.command("generate")
def generate(
    text: str = typer.Argument(..., help="Text to synthesize"),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Language code (default: kokoro_DEFAULT_LANGUAGE)",
    ),
    voice: str | None = typer.Option(
        None,
        "--voice",
        "-v",
        help="Voice identifier (optional, uses the configured default)",
    ),
    output: str = typer.Option(
        "output.wav",
        "--output",
        "-o",
        help="Output WAV file path",
    ),
):
    """Generate speech from text and save to WAV file."""
    settings = get_settings()
    language = language or settings.DEFAULT_LANGUAGE

    print("Generating speech...")
    print(f"  Text: {text[:50]}..." if len(text) > 50 else f"  Text: {text}")
    print(f"  Language: {language}")
    print(f"  Voice: {voice or settings.DEFAULT_VOICE}")

    # A CLI run serves a single language, so the backend is built for it.
    backend = build_backend(settings, lang_code=language)
    if not _load_backend(backend, advice=False):
        sys.exit(1)

    # Same application logic as the HTTP endpoint.
    service = SpeechService(backend, settings)

    try:
        audio_bytes, metadata = service.synthesize(
            {
                "input": text,
                "language": language,
                "voice": voice,
                "response_format": "wav",
            },
            output_file=output,
        )
    except KokoroTTSException as exc:
        logger.exception("Synthesis failed")
        print(f"\nERROR: Synthesis failed: {exc}")
        sys.exit(1)
    except Exception:  # CLI boundary: report, never a traceback
        logger.exception("Synthesis failed")
        print("\nERROR: Synthesis failed (see log for details)")
        sys.exit(1)

    print("\nSuccess!")
    print(f"  Output: {output}")
    print(f"  Sample rate: {metadata['sample_rate']} Hz")
    print(f"  Duration: ~{metadata['duration_seconds']:.2f} seconds")
    print(f"  Bytes: {len(audio_bytes)}")


@app.command("voices")
def voices(
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Only show voices for this language code",
    ),
):
    """List the voices that are available in this installation."""
    settings = get_settings()

    backend = build_backend(settings)
    if not _load_backend(backend, advice=False):
        sys.exit(1)

    available = backend.list_voices()

    if language:
        available = [v for v in available if v.get("language") == language]

    if not available:
        print("No voices available.")
        print("A voice is downloaded on first use; check the voice id and your connection.")
        return

    print("Available voices:")
    for entry in available:
        lang = entry.get("language") or "unknown"
        print(f"  {entry['id']} ({lang})")

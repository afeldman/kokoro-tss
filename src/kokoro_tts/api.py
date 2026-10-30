"""
FastAPI HTTP API endpoints for kokoro-tts.

The routers defined here are the only HTTP implementation of the service:
``kokoro_tts.app.create_app`` mounts them for the running server and the test
suite mounts exactly the same objects, so the tested path is the production
path.

Error responses are structured JSON. The success response of the speech
endpoint is binary WAV and never goes through JSON serialization.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from kokoro_tts.config import Settings, get_settings
from kokoro_tts.exceptions import (
    KokoroTTSException,
    LanguageNotFoundError,
    ModelNotLoadedError,
    TextValidationError,
    UnsupportedFormatError,
    VoiceNotFoundError,
)
from kokoro_tts.schemas import HealthResponse, SpeechRequest, VoicesResponse
from kokoro_tts.service import SpeechService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["audio"])
root_router = APIRouter(tags=["health"])

WAV_MEDIA_TYPE = "audio/wav"
WAV_FILENAME = "speech.wav"


class WavResponse(Response):
    """Binary WAV response, so OpenAPI advertises audio/wav and not JSON."""

    media_type = WAV_MEDIA_TYPE


def _sanitized_validation_errors(exc: RequestValidationError) -> list[dict]:
    """
    Strip the offending value from validation errors.

    FastAPI's default 422 body echoes the submitted value back under
    ``detail[*].input``, which reflects user text into the response (and its
    repr for a wrongly typed field). Only location, type and message are kept.
    """
    return [
        {
            "type": error.get("type"),
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg"),
        }
        for error in exc.errors()
    ]


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 422 with structured errors that carry no submitted content."""
    return JSONResponse(status_code=422, content={"detail": _sanitized_validation_errors(exc)})


def get_settings_dep() -> Settings:
    """Dependency to get settings."""
    return get_settings()


def get_backend(request: Request):
    """
    Dependency resolving the inference backend from application state.

    The server startup attaches the backend via ``app.state.backend``, which
    keeps the routes decoupled from backend construction and allows tests to
    override the backend through the application state.
    """
    backend = getattr(request.app.state, "backend", None)
    if backend is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference backend is not configured",
        )
    return backend


def get_service(request: Request) -> SpeechService:
    """
    Dependency resolving the shared speech service.

    Both HTTP and CLI reach the backend through ``SpeechService``; this is the
    only place the HTTP layer builds it.
    """
    backend = getattr(request.app.state, "backend", None)
    if backend is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference backend is not configured",
        )
    settings = getattr(request.app.state, "settings", None)
    return SpeechService(backend, settings)


def _health_response(backend) -> HealthResponse:
    """Build the health payload for a backend."""
    device = "unknown"
    if hasattr(backend, "device"):
        device = backend.device
    elif hasattr(backend, "_resolve_device"):
        device = backend._resolve_device()

    loaded = bool(backend.model_loaded)

    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
        device=device,
        backend="kokoro",
        language=getattr(backend, "language", None),
    )


@root_router.get("/health", response_model=HealthResponse, tags=["health"])
@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(backend: Annotated[Any, Depends(get_backend)]) -> HealthResponse:
    """
    Health check endpoint.

    The response always answers HTTP 200 when the process is up, so the two
    states stay distinguishable: ``status``/``model_loaded`` report whether the
    model is ready to serve, while reaching the endpoint at all proves the
    process is running.
    """
    return _health_response(backend)


@router.get("/audio/voices", response_model=VoicesResponse, tags=["audio"])
async def list_voices(backend: Annotated[Any, Depends(get_backend)]) -> VoicesResponse:
    """
    List available voices for the current language.

    Reports only voices that are actually loaded in this process and therefore
    synthesizable without a download.
    """
    if not backend.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded",
        )

    return VoicesResponse(voices=backend.list_voices())


@router.post(
    "/audio/speech",
    tags=["audio"],
    response_class=WavResponse,
    responses={
        200: {"content": {WAV_MEDIA_TYPE: {}}, "description": "Binary WAV audio"},
        400: {"description": "Invalid text or unavailable language"},
        404: {"description": "Unknown voice"},
        415: {"description": "Unsupported response format"},
        422: {"description": "Invalid request schema"},
        503: {"description": "Model not loaded"},
        500: {"description": "Internal error"},
    },
)
async def generate_speech(
    request: SpeechRequest,
    service: Annotated[SpeechService, Depends(get_service)],
) -> Response:
    """
    Synthesize speech from text and return it as binary WAV.

    Args:
        input: Text to synthesize (1 to kokoro_MAX_TEXT_LENGTH characters)
        language: Language tag (defaults to the instance language)
        voice: Optional voice identifier (defaults to the instance voice)
        response_format: Output format ('wav' currently supported)

    Returns:
        Binary WAV audio with ``Content-Type: audio/wav``

    Raises:
        400: Text validation error or unsupported language
        404: Unknown voice
        415: Unsupported response format
        422: Invalid request schema
        503: Model not loaded
        500: Inference failure (generic message, cause is logged server side)
    """
    if not service.backend.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet. Please restart the server.",
        )

    payload = request.model_dump()

    try:
        # Inference is CPU/GPU bound and holds a lock inside the backend:
        # run it in the threadpool so the event loop keeps serving requests.
        audio_bytes, metadata = await run_in_threadpool(service.synthesize, payload)

    except TextValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except LanguageNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except VoiceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except UnsupportedFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet. Please restart the server.",
        ) from exc

    except KokoroTTSException as exc:
        logger.exception(
            "Speech synthesis failed (voice=%s language=%s chars=%d)",
            payload.get("voice"),
            payload.get("language"),
            len(payload.get("input") or ""),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during speech generation",
        ) from exc

    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.exception(
            "Unexpected error during speech synthesis (voice=%s language=%s chars=%d)",
            payload.get("voice"),
            payload.get("language"),
            len(payload.get("input") or ""),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during speech generation",
        ) from exc

    logger.info(
        "Synthesized %d bytes of WAV (voice=%s language=%s chars=%d)",
        len(audio_bytes),
        metadata.get("voice"),
        metadata.get("language"),
        len(payload.get("input") or ""),
    )

    # Binary HTTP response. Returning the bytes inside a dict would make
    # FastAPI JSON-encode the payload and call bytes.decode() on the audio.
    # No response_model: this route is not JSON.
    return WavResponse(
        content=audio_bytes,
        headers={
            "Content-Disposition": f'attachment; filename="{WAV_FILENAME}"',
        },
    )

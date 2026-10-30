"""
Application factory for the kokoro-tts HTTP server.

``create_app`` is the single composition point of the HTTP layer: it wires the
tested router from :mod:`kokoro_tts.api` onto an application, with the
inference backend attached to the application state. The CLI ``serve`` command
uses this factory, so the server that runs in production serves exactly the
routes the test suite exercises.
"""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from kokoro_tts.api import request_validation_exception_handler, root_router, router
from kokoro_tts.config import Settings, get_settings


def create_app(backend, settings: Settings | None = None) -> FastAPI:
    """
    Build the HTTP application for a loaded inference backend.

    Args:
        backend: A loaded (or loading) ``InferenceBackend``
        settings: Optional settings; the process settings are used if omitted

    Returns:
        FastAPI application with ``backend`` exposed as ``app.state.backend``
    """
    app = FastAPI(
        title="Kokoro TTS API",
        description="Local Text-to-Speech Service using Kokoro",
        version="0.1.0",
    )

    app.state.backend = backend
    app.state.settings = settings or get_settings()

    # Validation errors must not echo the submitted value back to the caller.
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)

    app.include_router(root_router)
    app.include_router(router)

    return app

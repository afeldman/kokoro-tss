"""
ASGI entry point for the end-to-end test.

Serves the PRODUCTION application (``kokoro_tts.app.create_app``) with the fake
backend, so the end-to-end test exercises the real server path without
downloading model weights.

Run by the test suite as ``uvicorn tests.fake_server:app``.
"""

from kokoro_tts.app import create_app
from kokoro_tts.fake_backend import FakeInferenceBackend

backend = FakeInferenceBackend()
backend.load()

app = create_app(backend)

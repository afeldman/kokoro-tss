"""
HTTP contract tests for the kokoro-tts speech API.

The application under test is built by ``kokoro_tts.app.create_app`` - the same
factory ``kokoro-tts serve`` uses - so these tests exercise the production
router wiring and not a hand-built application.

Success responses of the speech endpoint are binary WAV and must never travel
through JSON serialization; error responses stay structured JSON.
"""

import asyncio
import io
import json
import struct
import time
import wave

import httpx
import pytest
from conftest import ExplodingBackend, RecordingFakeBackend
from fastapi.testclient import TestClient

from kokoro_tts.app import create_app
from kokoro_tts.audio import KOKORO_SAMPLE_RATE
from kokoro_tts.config import get_settings
from kokoro_tts.exceptions import InferenceError
from kokoro_tts.fake_backend import FakeInferenceBackend

SPEECH_URL = "/v1/audio/speech"

SPEECH_PAYLOAD = {
    "input": "Hello from Kokoro.",
    "language": "en-us",
    "voice": "af_heart",
    "response_format": "wav",
}


@pytest.fixture
def app(loaded_backend):
    """Production application with a loaded fake backend."""
    return create_app(loaded_backend)


@pytest.fixture
def client(app) -> TestClient:
    """
    Test client over the production application.

    ``raise_server_exceptions=False`` is deliberate: a defect in the response
    path surfaces as the HTTP status a real client would see, instead of as a
    test error.
    """
    return TestClient(app, raise_server_exceptions=False)


# --------------------------------------------------------------------------
# Wiring: the routes are the ones the server serves
# --------------------------------------------------------------------------


def test_production_routes_are_reachable(client):
    """Every documented endpoint answers on the application factory's app."""
    assert client.get("/health").status_code == 200
    assert client.get("/v1/health").status_code == 200
    assert client.get("/v1/audio/voices").status_code == 200
    assert client.post(SPEECH_URL, json=SPEECH_PAYLOAD).status_code == 200


# --------------------------------------------------------------------------
# GET /health - process running vs. model ready
# --------------------------------------------------------------------------


def test_health_reports_ready_model(client):
    """A loaded model reports healthy and the serving parameters."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["backend"] == "kokoro"
    assert body["language"] == "en-us"
    assert body["device"]


def test_health_distinguishes_running_process_from_ready_model():
    """Without a loaded model the process still answers, reporting not ready."""
    unloaded = FakeInferenceBackend()
    client = TestClient(create_app(unloaded), raise_server_exceptions=False)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["model_loaded"] is False


# --------------------------------------------------------------------------
# GET /v1/audio/voices
# --------------------------------------------------------------------------


def test_voices_lists_only_available_voices_with_metadata(client):
    """Voices come from the backend and carry a usable language tag."""
    response = client.get("/v1/audio/voices")

    assert response.status_code == 200
    voices = response.json()["voices"]
    assert [v["id"] for v in voices] == ["af_heart", "bm_george"]
    assert {v["id"]: v["language"] for v in voices} == {
        "af_heart": "en-us",
        "bm_george": "en-gb",
    }


def test_voices_are_empty_when_nothing_is_loaded():
    """An unloaded process does not advertise voices it cannot synthesize."""
    unloaded = FakeInferenceBackend()
    client = TestClient(create_app(unloaded), raise_server_exceptions=False)

    response = client.get("/v1/audio/voices")

    # The route refuses with 503 rather than publishing an empty inventory.
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded"


# --------------------------------------------------------------------------
# POST /v1/audio/speech - binary WAV success
# --------------------------------------------------------------------------


def test_speech_returns_binary_wav(client):
    response = client.post(SPEECH_URL, json=SPEECH_PAYLOAD)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["content-disposition"] == 'attachment; filename="speech.wav"'
    assert response.content[:4] == b"RIFF"
    assert response.content[8:12] == b"WAVE"


def test_speech_body_matches_backend_output_byte_for_byte(client, loaded_backend):
    """No JSON envelope, no base64, no text round-trip."""
    expected, _ = loaded_backend.synthesize(
        text=SPEECH_PAYLOAD["input"],
        language=SPEECH_PAYLOAD["language"],
        voice=SPEECH_PAYLOAD["voice"],
    )

    response = client.post(SPEECH_URL, json=SPEECH_PAYLOAD)

    assert response.status_code == 200
    assert response.content == expected
    assert struct.unpack("<I", response.content[4:8])[0] == len(response.content) - 8


def test_speech_body_is_valid_16bit_mono_pcm(client):
    """An independent stdlib parser accepts the delivered WAV container."""
    response = client.post(SPEECH_URL, json=SPEECH_PAYLOAD)

    assert response.status_code == 200
    with wave.open(io.BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == KOKORO_SAMPLE_RATE
        assert wav_file.getnframes() > 0


def test_speech_response_is_not_json(client):
    response = client.post(SPEECH_URL, json=SPEECH_PAYLOAD)

    assert response.status_code == 200
    assert "application/json" not in response.headers["content-type"]
    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        json.loads(response.content)


def test_speech_response_does_not_need_utf8_decoding(client):
    """The payload is binary; a strict UTF-8 decode must fail."""
    response = client.post(SPEECH_URL, json=SPEECH_PAYLOAD)

    assert response.status_code == 200
    with pytest.raises(UnicodeDecodeError):
        response.content.decode("utf-8")


# --------------------------------------------------------------------------
# POST /v1/audio/speech - rejected requests
# --------------------------------------------------------------------------


def test_speech_rejects_unsupported_format_with_415(client):
    response = client.post(SPEECH_URL, json={**SPEECH_PAYLOAD, "response_format": "mp3"})

    assert response.status_code == 415
    assert response.headers["content-type"].startswith("application/json")
    assert "mp3" in response.json()["detail"]


def test_speech_rejects_unknown_voice_with_404(client):
    response = client.post(SPEECH_URL, json={**SPEECH_PAYLOAD, "voice": "zz_unknown"})

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "zz_unknown" in response.json()["detail"]


@pytest.mark.parametrize("language", ["ja", "klingon"])
def test_speech_rejects_unavailable_language_with_400(client, language):
    """Known-but-not-enabled and unknown languages both yield a 4xx."""
    response = client.post(SPEECH_URL, json={**SPEECH_PAYLOAD, "language": language})

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert language in response.json()["detail"]


def test_speech_503_when_model_not_loaded():
    client = TestClient(create_app(FakeInferenceBackend()), raise_server_exceptions=False)

    response = client.post(SPEECH_URL, json=SPEECH_PAYLOAD)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded yet. Please restart the server."


def test_speech_503_when_backend_is_not_configured(app):
    app.state.backend = None
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(SPEECH_URL, json=SPEECH_PAYLOAD)

    assert response.status_code == 503
    assert response.json()["detail"] == "Inference backend is not configured"


def test_speech_rejects_empty_input_with_422(client):
    response = client.post(SPEECH_URL, json={**SPEECH_PAYLOAD, "input": ""})

    assert response.status_code == 422


def test_speech_rejects_whitespace_only_input_with_400(client):
    """Schema-valid but semantically empty text is refused before inference."""
    response = client.post(SPEECH_URL, json={**SPEECH_PAYLOAD, "input": "   "})

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_speech_rejects_overlong_input_with_422(client):
    limit = get_settings().MAX_TEXT_LENGTH

    response = client.post(SPEECH_URL, json={**SPEECH_PAYLOAD, "input": "x" * (limit + 1)})

    assert response.status_code == 422
    assert str(limit) in response.text


def test_max_text_length_follows_the_configuration(client, monkeypatch):
    """The boundary limit is read from the settings, not hard coded."""
    monkeypatch.setenv("kokoro_MAX_TEXT_LENGTH", "20")
    get_settings.cache_clear()
    try:
        too_long = client.post(SPEECH_URL, json={**SPEECH_PAYLOAD, "input": "x" * 21})
        allowed = client.post(SPEECH_URL, json={**SPEECH_PAYLOAD, "input": "x" * 20})
    finally:
        get_settings.cache_clear()

    assert too_long.status_code == 422
    assert allowed.status_code == 200


# --------------------------------------------------------------------------
# Internal failures must not leak upstream detail
# --------------------------------------------------------------------------


def test_validation_error_does_not_echo_the_submitted_text(client):
    """
    422 bodies carry error type, location and message - never the offending value.

    FastAPI's default validation body echoes the submitted input under
    ``detail[*].input``; this is the regression that guards the sanitising
    exception handler.
    """
    secret = "SUBMITTED-" + "s" * 6000

    response = client.post(SPEECH_URL, json={**SPEECH_PAYLOAD, "input": secret})

    assert response.status_code == 422
    assert "SUBMITTED-" not in response.text
    assert "input" in response.text  # the field name stays in `loc`
    assert len(response.text) < 500


def test_validation_error_hides_wrongly_typed_values(client):
    """A wrongly typed field must not leak its repr (paths, object contents)."""
    response = client.post(SPEECH_URL, json={"input": {"path": "/Users/someone/private/notes.txt"}})

    assert response.status_code == 422
    assert "/Users/" not in response.text
    assert "private" not in response.text


def test_openapi_advertises_binary_wav_for_the_speech_endpoint(app):
    """A generated client must not plan for a JSON body on this route."""
    content = app.openapi()["paths"][SPEECH_URL]["post"]["responses"]["200"]["content"]

    assert "audio/wav" in content


def test_internal_error_is_generic_and_leaks_nothing():
    """A backend failure surfaces as a generic 500 with a structured JSON body."""
    leaky = ExplodingBackend(
        InferenceError(
            "404 Client Error for url: https://huggingface.co/hexgrad/Kokoro-82M/"
            "resolve/main/voices/zz.pt?request_id=Root=1-6ad2adf9-55261c9b7dc73ef912e69736"
        )
    )
    leaky.load()
    client = TestClient(create_app(leaky), raise_server_exceptions=False)

    response = client.post(SPEECH_URL, json=SPEECH_PAYLOAD)

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")

    body = response.text
    assert response.json()["detail"] == "Internal server error during speech generation"
    for leak in ("huggingface", "http", "request_id", "Root=", "/Users/", ".pt", "voices/"):
        assert leak not in body


def test_unexpected_exception_is_generic_and_leaks_nothing():
    """Even a non-domain exception (e.g. a raw OSError) stays generic."""
    leaky = ExplodingBackend(
        OSError("/Users/anton.feldmann/.cache/huggingface/hub/models--x/blobs/abc not found")
    )
    leaky.load()
    client = TestClient(create_app(leaky), raise_server_exceptions=False)

    response = client.post(SPEECH_URL, json=SPEECH_PAYLOAD)

    assert response.status_code == 500
    body = response.text
    assert response.json()["detail"] == "Internal server error during speech generation"
    for leak in ("/Users/", ".cache", "huggingface", "blobs"):
        assert leak not in body


def test_error_response_does_not_echo_the_submitted_text(client):
    """Diagnostics use length and voice, never the submitted text."""
    secret = "MyVerySecretSentenceNobodyShouldLog"
    response = client.post(SPEECH_URL, json={**SPEECH_PAYLOAD, "input": secret, "voice": "zz_nope"})

    assert response.status_code == 404
    assert secret not in response.text


# --------------------------------------------------------------------------
# Concurrency: the event loop must stay responsive, requests must not mix
# --------------------------------------------------------------------------


def test_parallel_requests_are_all_served(loaded_backend):
    """Concurrent requests each receive their own complete WAV body."""
    app = create_app(loaded_backend)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            return await asyncio.gather(
                *[ac.post(SPEECH_URL, json=SPEECH_PAYLOAD) for _ in range(6)]
            )

    responses = asyncio.run(scenario())

    assert all(r.status_code == 200 for r in responses)
    for response in responses:
        assert response.headers["content-type"].startswith("audio/wav")
        assert response.content[:4] == b"RIFF"
        with wave.open(io.BytesIO(response.content), "rb") as wav_file:
            assert wav_file.getnframes() > 0


def test_event_loop_is_not_blocked_while_inference_runs(loaded_backend):
    """
    Inference runs in the threadpool, so other endpoints keep answering.

    A synchronous call on the event loop would block the health request for
    roughly the whole synthesis duration.
    """
    loaded_backend.delay_seconds = 0.8
    app = create_app(loaded_backend)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            speech = asyncio.create_task(ac.post(SPEECH_URL, json=SPEECH_PAYLOAD))
            await asyncio.sleep(0.15)
            started = time.monotonic()
            health = await ac.get("/health")
            elapsed = time.monotonic() - started
            speech_response = await speech
        return health, elapsed, speech_response

    health, elapsed, speech_response = asyncio.run(scenario())

    assert speech_response.status_code == 200
    assert health.status_code == 200
    assert elapsed < 0.5, f"health check was blocked for {elapsed:.3f}s during inference"


def test_synthesis_is_not_reinitialized_per_request(loaded_backend, client):
    """Serving requests never re-initializes the model."""
    assert loaded_backend.load_count == 1  # the fixture initialized it once

    for _ in range(3):
        assert client.post(SPEECH_URL, json=SPEECH_PAYLOAD).status_code == 200

    assert loaded_backend.load_count == 1  # unchanged by serving requests
    assert len(loaded_backend.calls) >= 3


def test_recording_backend_helper_is_loaded(loaded_backend):
    """Guard: the fixture used by the concurrency tests reports readiness."""
    assert isinstance(loaded_backend, RecordingFakeBackend)
    assert loaded_backend.model_loaded is True

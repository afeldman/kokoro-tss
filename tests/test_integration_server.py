"""
End-to-end test against a real server process.

Starts ``uvicorn tests.fake_server:app`` in a subprocess - the production
application factory with a fake backend, so no model weights are needed - and
talks to it over a real TCP socket with a real HTTP client.
"""

import io
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
import pytest

from kokoro_tts.audio import KOKORO_SAMPLE_RATE

REPO_ROOT = Path(__file__).resolve().parents[1]

SPEECH_PAYLOAD = {
    "input": "Hello from the end-to-end test.",
    "language": "en-us",
    "voice": "af_heart",
    "response_format": "wav",
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    """Run the production app in a real uvicorn process for this module."""
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.fake_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read().decode(errors="replace") if process.stdout else ""
            pytest.fail(f"server process exited before becoming ready:\n{output}")
        try:
            if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        process.terminate()
        pytest.fail("server did not become ready within 30s")

    yield url

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive teardown
        process.kill()
        process.wait(timeout=10)


def test_health_over_http(base_url):
    response = httpx.get(f"{base_url}/health", timeout=10)

    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["status"] == "healthy"
    assert body["backend"] == "kokoro"


def test_versioned_health_over_http(base_url):
    assert httpx.get(f"{base_url}/v1/health", timeout=10).status_code == 200


def test_voices_over_http(base_url):
    response = httpx.get(f"{base_url}/v1/audio/voices", timeout=10)

    assert response.status_code == 200
    voices = response.json()["voices"]
    assert voices and all(voice["id"] for voice in voices)
    assert all(voice["language"] for voice in voices)


def test_speech_over_http_returns_playable_wav(base_url):
    response = httpx.post(f"{base_url}/v1/audio/speech", json=SPEECH_PAYLOAD, timeout=30)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["content-disposition"] == 'attachment; filename="speech.wav"'
    assert response.content[:4] == b"RIFF"
    assert response.content[8:12] == b"WAVE"

    with wave.open(io.BytesIO(response.content), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == KOKORO_SAMPLE_RATE
        assert wav_file.getnframes() > 0

    assert len(response.content) == int(response.headers["content-length"])


def test_unsupported_format_over_http_is_rejected(base_url):
    response = httpx.post(
        f"{base_url}/v1/audio/speech",
        json={**SPEECH_PAYLOAD, "response_format": "mp3"},
        timeout=30,
    )

    assert response.status_code == 415
    assert response.headers["content-type"].startswith("application/json")
    assert "mp3" in response.json()["detail"]


def test_unknown_voice_over_http_is_rejected(base_url):
    response = httpx.post(
        f"{base_url}/v1/audio/speech",
        json={**SPEECH_PAYLOAD, "voice": "zz_ghost"},
        timeout=30,
    )

    assert response.status_code == 404
    assert "zz_ghost" in response.json()["detail"]


def test_unavailable_language_over_http_is_rejected(base_url):
    response = httpx.post(
        f"{base_url}/v1/audio/speech",
        json={**SPEECH_PAYLOAD, "language": "ja"},
        timeout=30,
    )

    assert response.status_code == 400
    assert "ja" in response.json()["detail"]


def test_invalid_schema_over_http_is_rejected(base_url):
    response = httpx.post(f"{base_url}/v1/audio/speech", json={"voice": "af_heart"}, timeout=30)

    assert response.status_code == 422


def test_empty_text_over_http_is_rejected(base_url):
    response = httpx.post(
        f"{base_url}/v1/audio/speech", json={**SPEECH_PAYLOAD, "input": "   "}, timeout=30
    )

    assert response.status_code == 400


def test_error_responses_leak_nothing(base_url):
    """No stack traces, paths, URLs or request ids in public error bodies."""
    response = httpx.post(
        f"{base_url}/v1/audio/speech",
        json={**SPEECH_PAYLOAD, "voice": "zz_ghost"},
        timeout=30,
    )

    lowered = response.text.lower()
    for leak in (
        "traceback",
        "/users/",
        "huggingface",
        "http://",
        "https://",
        "request_id",
        'file "',
    ):
        assert leak not in lowered

"""
Tests for the Typer CLI.

The important claim here is structural: ``kokoro-tts serve`` must serve the
production application from ``create_app`` (the same router the HTTP tests
exercise), and ``generate`` must go through the shared ``SpeechService`` rather
than a second implementation.
"""

import sys

import pytest
from conftest import RecordingFakeBackend
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from kokoro_tts import cli
from kokoro_tts.config import Settings, get_settings
from kokoro_tts.exceptions import InferenceError
from kokoro_tts.inference import normalize_language_code
from kokoro_tts.service import SpeechService

runner = CliRunner()

SPEECH_PAYLOAD = {
    "input": "Hello from the CLI test.",
    "language": "en-us",
    "voice": "af_heart",
    "response_format": "wav",
}


class FakeUvicorn:
    """Records what ``uvicorn.run`` would have been called with."""

    def __init__(self):
        self.calls = []

    def run(self, app, **kwargs):
        self.calls.append({"app": app, **kwargs})


@pytest.fixture
def cli_env(monkeypatch):
    """
    CLI environment with a fake backend and a captured uvicorn.

    Returns the list of backends the CLI built, so a test can inspect the one
    the command actually used.
    """
    monkeypatch.setenv("kokoro_MODEL_DEVICE", "cpu")
    get_settings.cache_clear()

    created: list[RecordingFakeBackend] = []

    def factory(**kwargs):
        backend = RecordingFakeBackend(
            lang_code=kwargs.get("lang_code", "en-us"),
            device=kwargs.get("device", "cpu"),
        )
        created.append(backend)
        return backend

    monkeypatch.setattr(cli, "InferenceBackend", factory)

    uvicorn = FakeUvicorn()
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)

    yield created, uvicorn
    get_settings.cache_clear()


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------


def test_serve_serves_the_production_application(cli_env):
    """The served app carries the backend in its state and answers the routes."""
    created, uvicorn = cli_env

    result = runner.invoke(cli.app, ["serve"])

    assert result.exit_code == 0, result.output
    assert len(uvicorn.calls) == 1

    backend = created[-1]
    assert backend.load_count == 1  # model initialized exactly once

    app = uvicorn.calls[0]["app"]
    assert app.state.backend is backend  # built by the application factory

    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/health").status_code == 200
    assert client.get("/v1/audio/voices").status_code == 200

    response = client.post("/v1/audio/speech", json=SPEECH_PAYLOAD)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")


def test_serve_binds_localhost_by_default(monkeypatch, cli_env):
    """Nothing is exposed beyond the local machine unless asked for."""
    monkeypatch.delenv("kokoro_HOST", raising=False)
    monkeypatch.delenv("kokoro_PORT", raising=False)
    get_settings.cache_clear()
    _, uvicorn = cli_env

    result = runner.invoke(cli.app, ["serve"])

    assert result.exit_code == 0
    call = uvicorn.calls[0]
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 8765


def test_serve_exits_when_the_model_cannot_be_loaded(monkeypatch, cli_env):
    class FailingBackend(RecordingFakeBackend):
        def load(self):
            raise InferenceError("Failed to load Kokoro model 'hexgrad/Kokoro-82M'")

    monkeypatch.setattr(cli, "InferenceBackend", lambda **kwargs: FailingBackend())

    result = runner.invoke(cli.app, ["serve"])

    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_serve_honours_settings_host_and_port(monkeypatch, cli_env):
    """The bind address defaults come from the settings, not from literals."""
    monkeypatch.setenv("kokoro_PORT", "9123")
    get_settings.cache_clear()
    _, uvicorn = cli_env

    result = runner.invoke(cli.app, ["serve"])

    assert result.exit_code == 0
    assert uvicorn.calls[0]["port"] == 9123


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------


def test_generate_writes_a_valid_wav(cli_env, tmp_path):
    output = tmp_path / "speech.wav"

    result = runner.invoke(cli.app, ["generate", "Hello world", "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert output.read_bytes()[:4] == b"RIFF"

    created, _ = cli_env
    assert created[-1].calls[-1]["text"] == "Hello world"


def test_generate_uses_the_configured_language_when_omitted(cli_env, tmp_path):
    output = tmp_path / "speech.wav"

    result = runner.invoke(cli.app, ["generate", "Hello", "--output", str(output)])

    assert result.exit_code == 0
    created, _ = cli_env
    assert created[-1].calls[-1]["language"] == "en-us"


def test_generate_fails_cleanly_for_an_unloadable_language(monkeypatch, cli_env, tmp_path):
    """An unsupported language stops the command with a message, not a traceback."""

    class StrictBackend(RecordingFakeBackend):
        def load(self):
            if normalize_language_code(self.lang_code) is None:
                raise InferenceError(f"Unsupported language: {self.lang_code}")
            super().load()

    monkeypatch.setattr(
        cli,
        "InferenceBackend",
        lambda **kwargs: StrictBackend(lang_code=kwargs.get("lang_code", "en-us")),
    )

    output = tmp_path / "speech.wav"
    result = runner.invoke(
        cli.app, ["generate", "Hello", "--language", "klingon", "--output", str(output)]
    )

    assert result.exit_code == 1
    assert not output.exists()
    assert "ERROR" in result.output


# --------------------------------------------------------------------------
# voices
# --------------------------------------------------------------------------


def test_voices_lists_available_voices_with_language(cli_env):
    result = runner.invoke(cli.app, ["voices"])

    assert result.exit_code == 0, result.output
    assert "af_heart" in result.output
    assert "en-us" in result.output


def test_voices_filters_by_language(cli_env):
    result = runner.invoke(cli.app, ["voices", "--language", "en-gb"])

    assert result.exit_code == 0
    assert "bm_george" in result.output
    assert "af_heart" not in result.output


# --------------------------------------------------------------------------
# Shared logic between CLI and HTTP
# --------------------------------------------------------------------------


def test_cli_and_http_go_through_the_same_service(cli_env, tmp_path, monkeypatch):
    """Both entry points call SpeechService.synthesize - one implementation."""
    _, uvicorn = cli_env
    seen: list[str] = []
    original = SpeechService.synthesize

    def spy(self, request_dict, output_file=None):
        seen.append("synthesize")
        return original(self, request_dict, output_file=output_file)

    monkeypatch.setattr(SpeechService, "synthesize", spy)

    # HTTP path (the app the serve command would run)
    runner.invoke(cli.app, ["serve"])
    app = uvicorn.calls[0]["app"]
    client = TestClient(app, raise_server_exceptions=False)
    http_response = client.post("/v1/audio/speech", json=SPEECH_PAYLOAD)
    calls_after_http = len(seen)

    # CLI path
    output = tmp_path / "speech.wav"
    cli_result = runner.invoke(cli.app, ["generate", "Hello", "--output", str(output)])

    assert http_response.status_code == 200
    assert cli_result.exit_code == 0
    assert calls_after_http >= 1
    assert len(seen) > calls_after_http, "the CLI did not go through SpeechService"
    assert output.read_bytes()[:4] == b"RIFF"


# --------------------------------------------------------------------------
# Safe defaults
# --------------------------------------------------------------------------


def test_settings_default_to_localhost():
    settings = Settings()

    assert settings.HOST == "127.0.0.1"
    assert settings.PORT == 8765


def test_serve_resolves_settings_at_call_time(monkeypatch, cli_env):
    """
    Environment settings apply to the process that starts, not to import time.

    An import-time literal would keep serving the values of whatever
    environment first imported the module.
    """
    monkeypatch.setenv("kokoro_HOST", "127.0.0.1")
    monkeypatch.setenv("kokoro_PORT", "9555")
    get_settings.cache_clear()
    _, uvicorn = cli_env

    result = runner.invoke(cli.app, ["serve"])

    assert result.exit_code == 0
    assert uvicorn.calls[0]["host"] == "127.0.0.1"
    assert uvicorn.calls[0]["port"] == 9555


def test_explicit_flags_override_settings(monkeypatch, cli_env):
    monkeypatch.setenv("kokoro_PORT", "9555")
    get_settings.cache_clear()
    _, uvicorn = cli_env

    result = runner.invoke(cli.app, ["serve", "--port", "9999"])

    assert result.exit_code == 0
    assert uvicorn.calls[0]["port"] == 9999

"""
Tests for the shared speech service.

The service is the single application logic both the HTTP API and the CLI go
through, so its validation and error mapping decide the behaviour of both.
"""

import pytest
from conftest import ExplodingBackend, RecordingFakeBackend

from kokoro_tts.audio import KOKORO_SAMPLE_RATE
from kokoro_tts.config import Settings
from kokoro_tts.exceptions import (
    InferenceError,
    LanguageNotFoundError,
    TextValidationError,
    UnsupportedFormatError,
    VoiceNotFoundError,
)
from kokoro_tts.service import SpeechService


@pytest.fixture
def backend() -> RecordingFakeBackend:
    """Loaded recording fake backend."""
    fake = RecordingFakeBackend()
    fake.load()
    return fake


def make_service(backend, **settings_kwargs) -> SpeechService:
    """Service with explicit settings (independent of the process environment)."""
    return SpeechService(backend, Settings(**settings_kwargs))


def test_service_writes_wav_file(backend, tmp_path):
    """The CLI path writes the encoded WAV to disk."""
    output = tmp_path / "nested" / "speech.wav"

    audio, metadata = make_service(backend).synthesize(
        {"input": "Hello world", "language": "en-us", "voice": "af_heart"},
        output_file=output,
    )

    assert output.read_bytes() == audio
    assert audio[:4] == b"RIFF"
    assert metadata["sample_rate"] == KOKORO_SAMPLE_RATE
    assert metadata["voice"] == "af_heart"


def test_service_strips_surrounding_whitespace(backend):
    make_service(backend).synthesize({"input": "  Hello  ", "language": "en-us"})

    assert backend.calls[-1]["text"] == "Hello"


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_service_rejects_empty_text(backend, text):
    with pytest.raises(TextValidationError):
        make_service(backend).synthesize({"input": text, "language": "en-us"})


def test_service_rejects_text_beyond_the_configured_limit(backend):
    service = make_service(backend, MAX_TEXT_LENGTH=10)

    with pytest.raises(TextValidationError) as excinfo:
        service.synthesize({"input": "x" * 11, "language": "en-us"})

    assert "10" in str(excinfo.value)
    assert backend.calls == []  # never reached the backend


def test_service_accepts_text_at_the_configured_limit(backend):
    service = make_service(backend, MAX_TEXT_LENGTH=10)

    audio, _ = service.synthesize({"input": "x" * 10, "language": "en-us"})

    assert audio[:4] == b"RIFF"


def test_service_rejects_unsupported_format(backend):
    with pytest.raises(UnsupportedFormatError) as excinfo:
        make_service(backend).synthesize(
            {"input": "Hi", "language": "en-us", "response_format": "mp3"}
        )

    assert "mp3" in str(excinfo.value)
    assert backend.calls == []


def test_service_uses_the_configured_default_language(backend):
    """An omitted language falls back to the configured default, not a literal."""
    service = make_service(backend, DEFAULT_LANGUAGE="en-us")

    service.synthesize({"input": "Hi"})

    assert backend.calls[-1]["language"] == "en-us"


def test_service_propagates_domain_errors_unchanged(backend):
    """Voice and language errors must reach the HTTP layer as themselves."""
    service = make_service(backend)

    with pytest.raises(VoiceNotFoundError):
        service.synthesize({"input": "Hi", "language": "en-us", "voice": "zz_nope"})

    with pytest.raises(LanguageNotFoundError):
        service.synthesize({"input": "Hi", "language": "klingon"})


def test_service_wraps_unexpected_errors_without_upstream_text():
    """An unknown failure becomes a generic InferenceError with the cause kept."""
    upstream = RuntimeError(
        "Connection error for url https://huggingface.co/hexgrad/Kokoro-82M (id=Root=1-abc)"
    )
    leaky = ExplodingBackend(upstream)
    leaky.load()

    with pytest.raises(InferenceError) as excinfo:
        make_service(leaky).synthesize({"input": "Hi", "language": "en-us"})

    assert str(excinfo.value) == "Speech synthesis failed"
    assert excinfo.value.__cause__ is upstream

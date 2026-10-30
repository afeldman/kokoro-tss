"""
Tests for the inference adapter.

These tests drive the PRODUCTION class ``InferenceBackend`` with a stand-in
KPipeline (no weights, no downloads): the lock that serializes inference, the
language and voice validation, the chunk handling and the idempotent ``load()``
are all properties of the real class, not of a test double.
"""

import sys
import threading
import time
import types

import numpy as np
import pytest

from kokoro_tts import exceptions
from kokoro_tts.audio import KOKORO_SAMPLE_RATE
from kokoro_tts.inference import (
    InferenceBackend,
    language_tag_for_code,
    language_tag_for_voice,
    normalize_language_code,
)


class _FakeTensor:
    """Minimal tensor stand-in supporting the access path used by the adapter."""

    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeResult:
    def __init__(self, array):
        self.audio = _FakeTensor(array)


class FakePipeline:
    """
    Stand-in for ``kokoro.KPipeline``.

    Yields ``chunks`` audio blocks per call and records concurrent entry, so a
    missing lock in the adapter shows up as an observable overlap.
    """

    def __init__(self, voices=("af_heart",), chunks=1, fail_voice=None, delay=0.05):
        self.voices = {voice: None for voice in voices}
        self.model = object()
        self.chunks = chunks
        self.fail_voice = fail_voice
        self.delay = delay
        self.calls = []
        self._active = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def load_voice(self, voice):
        if self.fail_voice is not None and voice == self.fail_voice:
            raise RuntimeError(
                "Entry Not Found for url: https://huggingface.co/hexgrad/Kokoro-82M/"
                f"resolve/main/voices/{voice}.pt"
            )
        self.voices[voice] = None

    def __call__(self, text, voice=None):
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            self.calls.append({"text": text, "voice": voice})
            if self.delay:
                time.sleep(self.delay)
            for _ in range(self.chunks):
                yield _FakeResult(np.full(100, 0.5, dtype=np.float32))
        finally:
            with self._lock:
                self._active -= 1


def ready_backend(pipeline=None, **kwargs) -> InferenceBackend:
    """Production backend with a stand-in pipeline instead of real weights."""
    backend = InferenceBackend(**kwargs)
    backend._pipeline = pipeline or FakePipeline()
    backend._loaded = True
    return backend


# --------------------------------------------------------------------------
# Language tables come from Kokoro itself
# --------------------------------------------------------------------------


def test_language_normalization_matches_installed_kokoro():
    """The accepted tags cannot drift from the installed model package."""
    from kokoro.pipeline import ALIASES

    for tag, alias in ALIASES.items():
        assert normalize_language_code(tag) == alias
        assert language_tag_for_code(alias) == tag
        assert normalize_language_code(tag.upper()) == alias


@pytest.mark.parametrize("value", ["klingon", "", None, "  "])
def test_unknown_languages_normalize_to_none(value):
    assert normalize_language_code(value) is None


@pytest.mark.parametrize(
    ("voice_id", "expected"),
    [
        ("af_heart", "en-us"),
        ("bm_george", "en-gb"),
        ("ja_MIZUKI", "ja"),
        # 'z' is a valid Kokoro code (Mandarin), so the prefix still resolves;
        # only a prefix outside Kokoro's table is unknown.
        ("zz_odd", "zh"),
        ("xx_odd", None),
        ("", None),
    ],
)
def test_voice_language_is_derived_from_the_id(voice_id, expected):
    assert language_tag_for_voice(voice_id) == expected


# --------------------------------------------------------------------------
# Model lifecycle
# --------------------------------------------------------------------------


def test_load_initializes_the_pipeline_exactly_once(monkeypatch):
    """Repeated load() calls must not build the model again."""
    built = []

    class CountingKPipeline:
        def __init__(self, lang_code=None, repo_id=None):
            built.append(lang_code)
            self.voices = {}
            self.model = types.SimpleNamespace(to=lambda device: self.model)

        def load_voice(self, voice):
            self.voices[voice] = None

    fake_kokoro = types.ModuleType("kokoro")
    fake_kokoro.KPipeline = CountingKPipeline
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)

    backend = InferenceBackend(lang_code="en-us", device="cpu")

    backend.load()
    backend.load()
    backend.load()

    assert built == ["en-us"]
    assert backend.model_loaded is True


def test_backend_reports_not_loaded_before_load():
    backend = InferenceBackend()

    assert backend.model_loaded is False
    assert backend.list_voices() == []


def test_backend_uses_the_shared_exception_classes():
    """
    The adapter must raise the exception classes the HTTP layer maps.

    A second, local definition of these classes would silently turn a 503 into
    a 500 because ``except`` would never match.
    """
    backend = InferenceBackend()

    with pytest.raises(exceptions.ModelNotLoadedError):
        backend.synthesize("Hello", "en-us")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_synthesize_accepts_the_instance_language():
    backend = ready_backend(lang_code="en-us")

    audio, metadata = backend.synthesize("Hello", "en-us")

    assert audio[:4] == b"RIFF"
    assert metadata["voice"] == "af_heart"
    assert metadata["language"] == "en-us"


@pytest.mark.parametrize("language", ["ja", "en-gb", "klingon"])
def test_synthesize_rejects_unavailable_languages(language):
    """A known Kokoro language that this instance does not serve is also refused."""
    pipeline = FakePipeline()
    backend = ready_backend(pipeline, lang_code="en-us")

    with pytest.raises(exceptions.LanguageNotFoundError) as excinfo:
        backend.synthesize("Hello", language)

    assert pipeline.calls == []  # nothing was synthesized
    assert "en-us" in str(excinfo.value)


def test_unknown_voice_raises_without_leaking_upstream_text():
    pipeline = FakePipeline(voices=("af_heart",), fail_voice="zz_ghost")
    backend = ready_backend(pipeline)

    with pytest.raises(exceptions.VoiceNotFoundError) as excinfo:
        backend.synthesize("Hello", "en-us", voice="zz_ghost")

    message = str(excinfo.value)
    assert "zz_ghost" in message
    assert "huggingface" not in message
    assert "http" not in message
    assert "Entry Not Found" not in message


def test_list_voices_reports_only_loaded_voices():
    pipeline = FakePipeline(voices=("af_heart", "bm_george"))
    backend = ready_backend(pipeline)

    voices = backend.list_voices()

    assert voices == [
        {"id": "af_heart", "language": "en-us"},
        {"id": "bm_george", "language": "en-gb"},
    ]


def test_load_voice_downloads_a_missing_voice_once():
    pipeline = FakePipeline(voices=("af_heart",))
    backend = ready_backend(pipeline)

    backend.load_voice("bm_george")
    backend.load_voice("bm_george")

    assert "bm_george" in pipeline.voices
    assert backend.list_voices() == [
        {"id": "af_heart", "language": "en-us"},
        {"id": "bm_george", "language": "en-gb"},
    ]


# --------------------------------------------------------------------------
# Synthesis behaviour
# --------------------------------------------------------------------------


def test_all_pipeline_chunks_are_concatenated():
    """A long text is chunked by the pipeline; every chunk belongs to the result."""
    pipeline = FakePipeline(chunks=3, delay=0.0)
    backend = ready_backend(pipeline)

    audio, metadata = backend.synthesize("A long text", "en-us")

    import io
    import wave

    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        assert wav_file.getnframes() == 300  # 3 chunks x 100 samples
    assert metadata["duration_seconds"] == pytest.approx(300 / KOKORO_SAMPLE_RATE, abs=1e-3)


def test_synthesis_serializes_concurrent_callers():
    """The adapter's lock must keep the model out of concurrent use."""
    pipeline = FakePipeline(chunks=1, delay=0.1)
    backend = ready_backend(pipeline)

    def worker():
        backend.synthesize("Hello", "en-us")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert pipeline.max_concurrent == 1, "concurrent model access was not prevented"


def test_empty_audio_is_reported_as_an_error():
    pipeline = FakePipeline(chunks=0, delay=0.0)
    backend = ready_backend(pipeline)

    with pytest.raises(exceptions.InferenceError):
        backend.synthesize("Hello", "en-us")


def test_pipeline_failure_becomes_a_generic_inference_error():
    class ExplodingPipeline(FakePipeline):
        def __call__(self, text, voice=None):
            raise RuntimeError(
                "curl error for https://huggingface.co/hexgrad/Kokoro-82M (request_id=Root=1-xyz)"
            )
            yield  # pragma: no cover - generator marker

    backend = ready_backend(ExplodingPipeline())

    with pytest.raises(exceptions.InferenceError) as excinfo:
        backend.synthesize("Hello", "en-us")

    message = str(excinfo.value)
    assert message == "Speech synthesis failed"
    assert "huggingface" not in message
    assert "curl" not in message


def test_explicit_device_is_not_silently_changed(monkeypatch):
    """A configured device wins over auto-detection."""
    backend = InferenceBackend(device="cpu")

    assert backend.device == "cpu"

    forced = InferenceBackend(device="mps")
    assert forced.device == "mps"

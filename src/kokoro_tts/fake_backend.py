"""
Fake inference backend for testing without model weights.

The double follows the contract of :class:`kokoro_tts.inference.InferenceBackend`:
``load()`` initializes once, ``model_loaded`` reports readiness, ``list_voices()``
reports what is actually available and ``synthesize()`` returns
``(wav_bytes, metadata)``. Keeping the shapes identical is what makes a test
failure meaningful - a fake that is merely convenient proves nothing about the
production path.
"""

import threading

import numpy as np

from kokoro_tts.audio import KOKORO_SAMPLE_RATE, encode_wav
from kokoro_tts.exceptions import (
    LanguageNotFoundError,
    ModelNotLoadedError,
    VoiceNotFoundError,
)
from kokoro_tts.inference import normalize_language_code

SAMPLE_RATE = KOKORO_SAMPLE_RATE
SAMPLE_DURATION_SECONDS = 0.1
DEFAULT_VOICES = (
    ("af_heart", "en-us"),
    ("bm_george", "en-gb"),
)


class FakeInferenceBackend:
    """
    Fake backend that simulates Kokoro inference without actual model loading.

    Used for testing API and CLI functionality without requiring the actual
    Kokoro model weights to be downloaded.
    """

    def __init__(
        self,
        lang_code: str = "en-us",
        device: str = "cpu",
        voices: dict | None = None,
    ):
        self.lang_code = lang_code
        self.device = device
        self._voices = dict(voices or DEFAULT_VOICES)
        self._lock = threading.RLock()
        self._loaded = False

    @property
    def language(self) -> str:
        """Language this backend instance serves."""
        return self.lang_code

    @property
    def model_loaded(self) -> bool:
        """Whether the (fake) model is initialized."""
        return self._loaded

    def load(self) -> None:
        """Initialize the fake model (idempotent, like the real adapter)."""
        with self._lock:
            self._loaded = True

    def list_voices(self) -> list[dict]:
        """Return the voices this fake backend holds."""
        if not self._loaded:
            return []
        return [{"id": voice_id, "language": lang} for voice_id, lang in self._voices.items()]

    def load_voice(self, voice: str) -> None:
        """Ensure a voice is present, mirroring the real adapter's errors."""
        if not self._loaded:
            raise ModelNotLoadedError("Model must be loaded before loading voices")
        if voice not in self._voices:
            raise VoiceNotFoundError(f"Voice is not available: {voice}")

    def synthesize(
        self,
        text: str,
        language: str,
        voice: str | None = None,
    ):
        """Generate fake WAV audio for testing, contract-identical to the real one."""
        with self._lock:
            if not self._loaded:
                raise ModelNotLoadedError("Model must be loaded before inference")

            # Same normalisation as the real adapter: the fake must not accept
            # or reject different language tags than production does.
            requested = normalize_language_code(language)
            served = normalize_language_code(self.lang_code)
            if requested is None or requested != served:
                raise LanguageNotFoundError(
                    f"Language is not available on this instance: {language} "
                    f"(instance language: {self.lang_code})"
                )

            voice_id = voice or next(iter(self._voices))
            self.load_voice(voice_id)

            samples = int(SAMPLE_RATE * SAMPLE_DURATION_SECONDS)
            timeline = np.linspace(0, SAMPLE_DURATION_SECONDS, samples, endpoint=False)
            audio = np.sin(2 * np.pi * 440 * timeline).astype(np.float32)

            wav_bytes = encode_wav(audio, sample_rate=SAMPLE_RATE)

            return wav_bytes, {
                "sample_rate": SAMPLE_RATE,
                "duration_seconds": SAMPLE_DURATION_SECONDS,
                "voice": voice_id,
                "language": self.lang_code,
            }

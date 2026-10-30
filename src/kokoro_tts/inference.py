"""
Inference adapter for Kokoro TTS backend.

This module provides a unified interface to the Kokoro model, abstracting
away the concrete implementation details (KPipeline, G2P, tokenization).

The adapter supports:
- Model initialization once per process, with device management (MPS/CPU)
- Voice loading and management (lazily downloaded from the Hugging Face cache)
- Text-to-speech inference via the high level KPipeline API, which performs
  grapheme-to-phoneme conversion and chunking of long texts

Thread safety is ensured via a reentrant lock held for the whole inference
operation, so the model is never entered concurrently.
"""

import threading
from pathlib import Path

from kokoro_tts.exceptions import (
    InferenceError,
    LanguageNotFoundError,
    ModelNotLoadedError,
    VoiceNotFoundError,
)

DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"
DEFAULT_VOICE = "af_heart"


def normalize_language_code(language: str | None) -> str | None:
    """
    Map a user supplied language tag to Kokoro's internal single letter code.

    The accepted set is read from Kokoro's own ``ALIASES`` / ``LANG_CODES``
    tables, so it cannot drift away from the installed model package.
    Returns ``None`` for tags Kokoro does not know.
    """
    if not language:
        return None

    from kokoro.pipeline import ALIASES, LANG_CODES

    key = language.strip().lower()
    alias = ALIASES.get(key, key)
    return alias if alias in LANG_CODES else None


def language_tag_for_code(code: str) -> str | None:
    """Reverse lookup: Kokoro's single letter code -> public language tag."""
    from kokoro.pipeline import ALIASES, LANG_CODES

    if code not in LANG_CODES:
        return None

    for tag, alias in ALIASES.items():
        if alias == code:
            return tag
    return None


def language_tag_for_voice(voice_id: str) -> str | None:
    """
    Derive the language of a Kokoro voice.

    Voice ids follow ``<language letter><gender>_<name>`` (e.g. ``af_heart``,
    ``bm_george``), so the leading letter carries the language.
    """
    if not voice_id:
        return None
    return language_tag_for_code(voice_id[0])


class InferenceBackend:
    """
    Kokoro inference backend adapter.

    Encapsulates the Kokoro KPipeline and provides a consistent interface
    for text-to-speech synthesis with thread-safe concurrent access.

    The model is loaded exactly once per process via :meth:`load` (called
    during server startup), never per request.
    """

    def __init__(
        self,
        lang_code: str = "en-us",
        repo_id: str | None = None,
        device: str = "auto",
        cache_dir: str | Path = "~/.cache/kokoro",
        default_voice: str = DEFAULT_VOICE,
    ):
        """
        Initialize the inference backend.

        Args:
            lang_code: Language code (e.g., 'en-us', 'en-gb', 'es')
            repo_id: HuggingFace model repository ID, defaults to hexgrad/Kokoro-82M
            device: Device to run on ('auto', 'mps', 'cpu')
            cache_dir: Directory for cached models and voices
            default_voice: Voice used when a request does not name one
        """
        self.lang_code = lang_code
        self.repo_id = repo_id or DEFAULT_REPO_ID
        self.cache_dir = Path(cache_dir).expanduser()
        self.default_voice = default_voice

        # Thread safety lock for inference operations
        self._lock = threading.RLock()

        # Model instance - None until loaded
        self._pipeline = None
        self._loaded = False
        self._device = device

    @property
    def language(self) -> str:
        """Language tag this backend instance is configured for."""
        return self.lang_code

    @property
    def language_alias(self) -> str | None:
        """Kokoro's internal single letter code for this instance's language."""
        return normalize_language_code(self.lang_code)

    def load(self) -> None:
        """
        Load the Kokoro model.

        Called once during server startup. Idempotent: a second call is a
        no-op, so the weights are never initialized twice per process.
        """
        with self._lock:
            if self._loaded:
                return

            try:
                import kokoro

                pipeline = kokoro.KPipeline(
                    lang_code=self.lang_code,
                    repo_id=self.repo_id,
                )

                # Load the default voice eagerly so the first request does not
                # have to reach the network for it.
                pipeline.load_voice(self.default_voice)

                # Set device for model weights
                device_str = self._resolve_device()
                model = getattr(pipeline, "model", None)
                if model is not None and hasattr(model, "to"):
                    model.to(device_str)

                # Ensure the declared cache directory exists
                (self.cache_dir / "voices").mkdir(parents=True, exist_ok=True)

                self._pipeline = pipeline
                self._loaded = True
            except Exception as exc:
                self._pipeline = None
                self._loaded = False
                raise InferenceError(
                    f"Failed to load Kokoro model '{self.repo_id}': {exc}"
                ) from exc

    @property
    def model_loaded(self) -> bool:
        """Check if the model is loaded and usable."""
        return self._loaded and getattr(self._pipeline, "model", None) is not None

    @property
    def device(self) -> str:
        """Get the device string for the model."""
        return self._resolve_device()

    def _resolve_device(self) -> str:
        """
        Resolve device from configuration, preferring MPS on Apple Silicon.

        An explicitly configured device ('mps' or 'cpu') is never overridden.
        """
        if self._device == "auto":
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self._device

    def list_voices(self) -> list[dict]:
        """
        List the voices that are actually available in this process.

        Only voices already loaded into the pipeline are reported: an id that
        appears here can be synthesized immediately, without any download.
        """
        with self._lock:
            pipeline = self._pipeline
            if not self._loaded or pipeline is None:
                return []

            return [
                {"id": voice_id, "language": language_tag_for_voice(voice_id)}
                for voice_id in pipeline.voices
            ]

    def load_voice(self, voice: str) -> None:
        """
        Make sure a voice is available, downloading it into the HF cache once.

        Raises:
            ModelNotLoadedError: If the model itself is not loaded
            VoiceNotFoundError: If the voice cannot be resolved
        """
        with self._lock:
            pipeline = self._pipeline
            if not self._loaded or pipeline is None:
                raise ModelNotLoadedError("Model must be loaded before loading voices")

            if voice in pipeline.voices:
                return

            try:
                pipeline.load_voice(voice)
            except Exception as exc:
                # Deliberately not surfaces the upstream text: the cause stays
                # on the exception chain for the server side log.
                raise VoiceNotFoundError(f"Voice is not available: {voice}") from exc

    def synthesize(
        self,
        text: str,
        language: str,
        voice: str | None = None,
    ):
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            language: Language tag for the text; must match this instance
            voice: Optional voice identifier (defaults to the configured voice)

        Returns:
            Tuple of (audio_bytes as WAV, metadata dict)

        Raises:
            ModelNotLoadedError: If the model is not loaded
            LanguageNotFoundError: If the language is unknown or not enabled here
            VoiceNotFoundError: If the requested voice is not available
            InferenceError: If inference fails
        """
        with self._lock:
            if not self._loaded:
                raise ModelNotLoadedError("Model must be loaded before inference")

            requested = normalize_language_code(language)
            if requested is None or requested != self.language_alias:
                raise LanguageNotFoundError(
                    f"Language is not available on this instance: {language} "
                    f"(instance language: {self.lang_code})"
                )

            voice_id = voice or self.default_voice
            self.load_voice(voice_id)

            audio_np = self._generate_audio(text, voice_id)

            from kokoro_tts.audio import KOKORO_SAMPLE_RATE, encode_wav

            wav_bytes = encode_wav(audio_np, sample_rate=KOKORO_SAMPLE_RATE)

            return wav_bytes, {
                "sample_rate": KOKORO_SAMPLE_RATE,
                "duration_seconds": round(len(audio_np) / KOKORO_SAMPLE_RATE, 3),
                "voice": voice_id,
                "language": self.lang_code,
            }

    def _generate_audio(self, text: str, voice_id: str):
        """
        Run the pipeline and concatenate all audio chunks it yields.

        The high level ``KPipeline.__call__`` performs grapheme-to-phoneme
        conversion and splits long texts into model sized chunks; every chunk
        it yields belongs to the requested text and must be kept.
        """
        import numpy as np

        pipeline = self._pipeline
        if pipeline is None:
            raise ModelNotLoadedError("Model must be loaded before inference")

        chunks = []
        try:
            for result in pipeline(text, voice=voice_id):
                audio = result.audio
                if audio is None:
                    continue
                chunks.append(np.asarray(audio.detach().cpu().numpy()).reshape(-1))
        except Exception as exc:
            raise InferenceError("Speech synthesis failed") from exc

        if not chunks:
            raise InferenceError("Speech synthesis produced no audio data")

        audio_np = np.concatenate(chunks).astype(np.float32)

        # Normalize to [-1, 1]
        max_val = float(abs(audio_np).max())
        if max_val > 0 and max_val < float("inf"):
            audio_np = audio_np / max_val

        return audio_np

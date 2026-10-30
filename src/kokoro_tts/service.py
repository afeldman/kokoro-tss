"""
Shared speech service for HTTP API and CLI.

The service is the single place where a synthesis request is validated and
handed to the inference backend. Both the CLI and the HTTP API go through
this class, so there is exactly one implementation of the application logic.
"""

from pathlib import Path

from kokoro_tts.config import Settings, get_settings
from kokoro_tts.exceptions import (
    InferenceError,
    KokoroTTSException,
    TextValidationError,
    UnsupportedFormatError,
)

SUPPORTED_RESPONSE_FORMATS = ("wav",)


class SpeechService:
    """
    Shared speech synthesis service.

    Provides a unified interface for text-to-speech synthesis that is
    used by both the HTTP API and the CLI commands.
    """

    def __init__(self, backend, settings: Settings | None = None):
        """
        Initialize the speech service.

        Args:
            backend: The InferenceBackend instance for model access
            settings: Optional settings; the process settings are used if omitted
        """
        self.backend = backend
        self._settings = settings

    @property
    def settings(self) -> Settings:
        """Effective settings for this service."""
        return self._settings or get_settings()

    def synthesize(self, request_dict, output_file=None):
        """
        Synthesize speech from a text request.

        Args:
            request_dict: Dict with input, language, voice, response_format keys
            output_file: Optional file path to write WAV output (CLI use)

        Returns:
            Tuple of (audio_bytes as bytes, metadata dict)

        Raises:
            TextValidationError: If text is empty or too long
            UnsupportedFormatError: If the response format is not supported
            VoiceNotFoundError: If requested voice is not available
            LanguageNotFoundError: If the language is not available here
            ModelNotLoadedError: If the model is not loaded
            InferenceError: If inference fails
        """
        text = (request_dict.get("input") or "").strip()
        if not text:
            raise TextValidationError("Input text must not be empty")

        max_length = self.settings.MAX_TEXT_LENGTH
        if len(text) > max_length:
            raise TextValidationError(f"Input text must not exceed {max_length} characters")

        response_format = request_dict.get("response_format") or SUPPORTED_RESPONSE_FORMATS[0]
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise UnsupportedFormatError(
                f"Unsupported response format: {response_format}. "
                f"Supported formats: {', '.join(SUPPORTED_RESPONSE_FORMATS)}"
            )

        language = request_dict.get("language") or self.settings.DEFAULT_LANGUAGE
        voice = request_dict.get("voice") or None

        try:
            # The backend raises the domain exceptions itself (missing model,
            # unknown voice, unsupported language); anything else is an
            # unexpected internal failure and must not leak its message.
            audio_bytes, metadata = self.backend.synthesize(
                text=text,
                language=language,
                voice=voice,
            )
        except KokoroTTSException:
            raise
        except Exception as exc:
            raise InferenceError("Speech synthesis failed") from exc

        if output_file is not None:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)

        return audio_bytes, metadata

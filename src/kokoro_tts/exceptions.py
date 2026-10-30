"""
Exceptions module for kokoro-tts.
"""


class KokoroTTSException(Exception):
    """Base exception for kokoro-tts."""


class ModelNotLoadedError(KokoroTTSException):
    """Raised when the model is not loaded yet."""


class VoiceNotFoundError(KokoroTTSException):
    """Raised when a requested voice is not available."""


class LanguageNotFoundError(KokoroTTSException):
    """Raised when a requested language is not supported."""


class TextValidationError(KokoroTTSException):
    """Raised for text validation errors (empty, too long)."""


class UnsupportedFormatError(KokoroTTSException):
    """Raised for unsupported output formats."""


class InferenceError(KokoroTTSException):
    """Raised for inference-related errors."""

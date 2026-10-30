"""
Configuration module for kokoro-tts.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import pydantic_settings


class Settings(pydantic_settings.BaseSettings):
    """Application settings loaded from environment variables."""

    # Server configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8765
    LOG_LEVEL: str = "INFO"

    # Model configuration
    MODEL_ID: str | None = "hexgrad/Kokoro-82M"
    MODEL_DEVICE: Literal["auto", "mps", "cpu"] = "auto"
    MODEL_CACHE_DIR: str = "~/.cache/kokoro"

    # Voice and language defaults
    DEFAULT_VOICE: str = "af_heart"
    DEFAULT_LANGUAGE: str = "en-us"

    # Validation limits
    MAX_TEXT_LENGTH: int = 5000

    # Request handling
    REQUEST_TIMEOUT_SECONDS: int = 60

    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env",
        env_prefix="kokoro_",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get settings instance (cached)."""
    return Settings()


def parse_device(device_str: str) -> str:
    """
    Parse device string, handling 'auto' detection.

    On Apple Silicon with PyTorch installed, prefers MPS if available.
    Falls back to CPU otherwise.
    """
    if device_str == "auto":
        import torch

        # Check for MPS availability (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        # Fallback to CPU
        return "cpu"

    return device_str


def parse_model_cache_dir(cache_dir: str) -> Path:
    """Resolve model cache directory path."""
    expanded = Path(cache_dir).expanduser()
    expanded.mkdir(parents=True, exist_ok=True)
    return expanded

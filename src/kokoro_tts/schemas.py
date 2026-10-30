"""
Request and Response models for kokoro-tts.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kokoro_tts.config import get_settings


class SpeechRequest(BaseModel):
    """Request model for text-to-speech synthesis."""

    input: str = Field(
        ...,
        min_length=1,
        description="Text to synthesize (1 to kokoro_MAX_TEXT_LENGTH characters)",
    )
    language: str | None = Field(
        default=None,
        description=(
            "Language tag for phonemization (e.g. 'en-us'). Defaults to "
            "kokoro_DEFAULT_LANGUAGE of the running instance."
        ),
    )
    voice: str | None = Field(
        default=None,
        description="Optional voice identifier (defaults to the server's voice)",
    )
    response_format: str = Field(
        default="wav",
        description="Output format ('wav' currently supported)",
    )

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("input")
    @classmethod
    def _validate_input_length(cls, value: str) -> str:
        """
        Reject over-long text at the request boundary.

        The limit is read from the settings so the API contract and the
        configured maximum cannot drift apart.
        """
        limit = get_settings().MAX_TEXT_LENGTH
        if len(value) > limit:
            raise ValueError(f"Input text must not exceed {limit} characters")
        return value


class VoiceInfo(BaseModel):
    """Information about a single voice."""

    id: str = Field(..., description="Voice identifier")
    language: str | None = Field(None, description="Language supported")
    name: str | None = Field(None, description="Display name if available")


class VoicesResponse(BaseModel):
    """Response listing available voices."""

    voices: list[VoiceInfo] = Field(..., description="List of available voices")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status ('healthy' or 'unhealthy')")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Device used for inference ('mps', 'cpu')")
    backend: str = Field(..., description="Inference backend name")
    language: str | None = Field(None, description="Language this instance was started with")

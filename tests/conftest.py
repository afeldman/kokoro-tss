"""Pytest configuration and fixtures."""

import threading
import time

import pytest

from kokoro_tts.config import get_settings
from kokoro_tts.fake_backend import FakeInferenceBackend


class RecordingFakeBackend(FakeInferenceBackend):
    """
    Fake backend that records what the application layer asked for.

    Adds the observability the production backend does not need: how often the
    model was initialized, which requests went through, and how long a single
    synthesis may be held up so the event loop can be probed while inference
    runs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_calls: list[float] = []
        self.calls: list[dict] = []
        self.delay_seconds = 0.0
        self._active = 0
        self.max_concurrent = 0
        self._counter_lock = threading.Lock()

    def load(self) -> None:
        self.load_calls.append(time.monotonic())
        super().load()

    @property
    def load_count(self) -> int:
        """How often the model was initialized."""
        return len(self.load_calls)

    def synthesize(self, text: str, language: str, voice=None):
        with self._counter_lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            self.calls.append({"text": text, "language": language, "voice": voice})
            return super().synthesize(text=text, language=language, voice=voice)
        finally:
            with self._counter_lock:
                self._active -= 1


class ExplodingBackend(FakeInferenceBackend):
    """Fake backend whose synthesis fails with a given exception."""

    def __init__(self, error: Exception, **kwargs):
        super().__init__(**kwargs)
        self._error = error

    def synthesize(self, text: str, language: str, voice=None):
        raise self._error


@pytest.fixture
def fake_backend() -> FakeInferenceBackend:
    """Unloaded fake backend (mirrors a process that has not initialized yet)."""
    return FakeInferenceBackend()


@pytest.fixture
def loaded_backend() -> RecordingFakeBackend:
    """Loaded recording fake backend."""
    backend = RecordingFakeBackend()
    backend.load()
    return backend


@pytest.fixture
def settings():
    """Effective process settings."""
    return get_settings()

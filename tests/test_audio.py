"""Test audio encoding module."""

import struct

import numpy as np

from kokoro_tts.audio import BITS_PER_SAMPLE, KOKORO_SAMPLE_RATE, encode_wav


def test_encode_wav_basic():
    """Test basic WAV encoding with float32 input."""
    # Test signal: sine wave at 440Hz (A4)
    sample_rate = 22050
    duration = 0.5  # seconds
    frequency = 440  # Hz
    samples = int(sample_rate * duration)

    t = np.linspace(0, duration, samples, endpoint=False)
    audio = np.sin(2 * np.pi * frequency * t)

    # Encode to WAV
    wav_bytes = encode_wav(audio, sample_rate=sample_rate)

    assert len(wav_bytes) > 0, "WAV bytes should not be empty"
    assert isinstance(wav_bytes, bytes), "Should return bytes"

    # Verify WAV header structure (RIFF/WAVE/fmt/data)
    assert wav_bytes[:4] == b"RIFF", "RIFF header missing"
    assert wav_bytes[8:12] == b"WAVE", "WAVE format missing"

    # Verify file size matches expected (RIFF header says total - 8)
    riff_chunk_size = struct.unpack("<I", wav_bytes[4:8])[0]
    expected_file_size = riff_chunk_size + 8
    assert len(wav_bytes) == expected_file_size, (
        f"File size mismatch: {len(wav_bytes)} != {expected_file_size}"
    )


def test_encode_wav_header_fields_are_consistent():
    """
    Every derived header field agrees with rate, channels and bit depth.

    ``blockAlign`` was previously written as 16 instead of channels * bits / 8,
    which made the header internally inconsistent.
    """
    raw = encode_wav(np.zeros(100, dtype=np.float32), sample_rate=24000)

    channels = struct.unpack("<H", raw[22:24])[0]
    sample_rate = struct.unpack("<I", raw[24:28])[0]
    byte_rate = struct.unpack("<I", raw[28:32])[0]
    block_align = struct.unpack("<H", raw[32:34])[0]
    bits = struct.unpack("<H", raw[34:36])[0]
    data_size = struct.unpack("<I", raw[40:44])[0]

    assert channels == 1
    assert sample_rate == 24000
    assert bits == BITS_PER_SAMPLE
    assert block_align == channels * bits // 8 == 2
    assert byte_rate == sample_rate * block_align
    assert data_size == len(raw) - 44 == 200


def test_encode_wav_default_rate_is_the_model_rate():
    """The default rate is Kokoro's 24 kHz, not a placeholder."""
    raw = encode_wav(np.zeros(10, dtype=np.float32))

    assert KOKORO_SAMPLE_RATE == 24000
    assert struct.unpack("<I", raw[24:28])[0] == KOKORO_SAMPLE_RATE


def test_encode_wav_empty_header_is_consistent():
    """The empty-input header is built by the same code path as the normal one."""
    raw = encode_wav(np.array([], dtype=np.float32))

    assert len(raw) == 44
    assert struct.unpack("<H", raw[32:34])[0] == 2  # block align
    assert struct.unpack("<I", raw[40:44])[0] == 0  # no data


def test_encode_wav_normalization():
    """Audio outside [-1, 1] is scaled to full scale, not clipped or dropped."""
    audio = np.array([2.0, 3.0, -2.0, -3.0], dtype=np.float32)

    raw = encode_wav(audio)
    samples = np.frombuffer(raw[44:], dtype="<i2")

    assert len(samples) == 4
    assert samples.max() == 32767  # 3.0 is the peak -> full scale
    assert samples.min() == -32767  # and so is -3.0
    assert abs(int(samples[0]) - round(2 / 3 * 32767)) <= 1  # 2.0 -> 2/3 scale


def test_encode_wav_monotone():
    """The requested sample rate reaches the header, and the data length matches."""
    sample_rate = 11025
    duration = 0.1
    samples = int(sample_rate * duration)
    audio = np.zeros(samples, dtype=np.float32)
    mid_point = samples // 2
    audio[mid_point : mid_point + 10] = [float(i) / 10.0 for i in range(10)]

    raw = encode_wav(audio, sample_rate=sample_rate)

    assert struct.unpack("<I", raw[24:28])[0] == sample_rate
    assert struct.unpack("<I", raw[40:44])[0] == samples * 2
    assert len(raw) == 44 + samples * 2

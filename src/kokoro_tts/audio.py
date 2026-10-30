"""
Audio encoding module for WAV file creation.
"""

import struct

import numpy as np

# Kokoro-82M (hexgrad/Kokoro-82M) synthesizes at 24 kHz. This is measured against
# the model's own time base rather than the header we write: the pipeline advances
# 600 audio samples per pred_dur frame and documents 40 frames per second
# (kokoro/pipeline.py, ``join_timestamps``), so samples / sum(pred_dur) = 600.
# Labelling the data as 22050 Hz plays it back ~8.8 % too slow and ~1.5 semitones
# too low, which is why this constant is the single source of the rate.
KOKORO_SAMPLE_RATE = 24000

BITS_PER_SAMPLE = 16


def _wav_header(data_size: int, sample_rate: int, num_channels: int) -> bytes:
    """
    Build the canonical 44-byte PCM RIFF/WAVE header for a data chunk.

    All header fields (byte rate, block align) are derived from the sample rate,
    the channel count and the bit depth, so they cannot disagree with each other.
    """
    bytes_per_sample = BITS_PER_SAMPLE // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align

    header = bytearray()
    header.extend(b"RIFF")
    header.extend(struct.pack("<I", 36 + data_size))  # file size - 8
    header.extend(b"WAVE")

    # fmt subchunk
    header.extend(b"fmt ")
    header.extend(struct.pack("<I", 16))  # subchunk size
    header.extend(struct.pack("<H", 1))  # audio format: PCM
    header.extend(struct.pack("<H", num_channels))
    header.extend(struct.pack("<I", sample_rate))
    header.extend(struct.pack("<I", byte_rate))
    header.extend(struct.pack("<H", block_align))
    header.extend(struct.pack("<H", BITS_PER_SAMPLE))

    # data subchunk
    header.extend(b"data")
    header.extend(struct.pack("<I", data_size))

    return bytes(header)


def encode_wav(
    audio: np.ndarray | list[float],
    sample_rate: int = KOKORO_SAMPLE_RATE,
    num_channels: int = 1,
) -> bytes:
    """
    Encode audio samples to WAV format (16-bit PCM).

    Args:
        audio: Audio samples in range [-1, 1] as numpy array or list
        sample_rate: Sample rate in Hz (default: Kokoro's 24000 Hz)
        num_channels: Number of channels (1 = mono)

    Returns:
        WAV-encoded audio as bytes
    """
    # Convert to numpy if needed
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)

    # Handle empty array: a valid header without audio data
    if len(audio) == 0:
        return _wav_header(0, sample_rate, num_channels)

    # Normalize to [-1, 1] range
    max_val = float(abs(audio).max())
    if max_val > 0 and max_val < float("inf"):
        audio = audio / max_val

    # Convert to 16-bit PCM
    data = (audio * 32767).astype(np.int16).tobytes()

    return _wav_header(len(data), sample_rate, num_channels) + data

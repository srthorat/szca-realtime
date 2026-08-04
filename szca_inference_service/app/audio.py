"""Audio I/O helpers: decode uploaded WAV bytes and resample for STT."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV/FLAC/OGG bytes to mono float32 + sample rate."""
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = wav.mean(axis=1)  # downmix to mono
    return mono.astype(np.float32), int(sr)


def resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear resample. Adequate for 16k STT ingestion of speech."""
    if src_sr == dst_sr:
        return wav.astype(np.float32)
    n_out = int(round(len(wav) * dst_sr / src_sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def encode_wav(wav: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float waveform to 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()

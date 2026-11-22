"""Audio file loading and resampling (no Qt)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".w64", ".caf")


def is_audio_file(path) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def load_audio(path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Read *path* as float32 ``(samples, channels)``; resampled to *sr* if given.

    soundfile (libsndfile) reads WAV/FLAC/OGG/MP3/AIFF; librosa is tried as a
    fallback for anything else.
    """
    try:
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 — unsupported by libsndfile: try librosa
        import librosa
        y, rate = librosa.load(str(path), sr=None, mono=False)
        data = np.atleast_2d(y).T.astype(np.float32)
    if sr is not None and int(rate) != int(sr):
        data = resample(data, int(rate), int(sr))
        rate = int(sr)
    return np.ascontiguousarray(data, dtype=np.float32), int(rate)


def resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """soxr (fastest) or polyphase (scipy), with a linear-interpolation fallback."""
    if sr_in == sr_out or len(data) == 0:
        return data
    try:
        import soxr
        return soxr.resample(np.ascontiguousarray(data, dtype=np.float32), sr_in, sr_out,
                             quality="HQ").astype(np.float32, copy=False)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly
        f = Fraction(sr_out, sr_in)            # exact ratio: no drift on long files
        return resample_poly(data, f.numerator, f.denominator, axis=0).astype(np.float32)
    except ImportError:
        n_out = int(round(len(data) * sr_out / sr_in))
        x_old = np.arange(len(data)) / sr_in
        x_new = np.arange(n_out) / sr_out
        return np.stack([np.interp(x_new, x_old, data[:, c]) for c in range(data.shape[1])],
                        axis=1).astype(np.float32)


def frames(y: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    """``(frame_length, n_frames)`` strided view, like ``librosa.util.frame``."""
    if len(y) < frame_length:
        return np.zeros((frame_length, 0), dtype=y.dtype)
    return np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop].T

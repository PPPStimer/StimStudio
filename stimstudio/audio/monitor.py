"""Signal measurements for the output monitor: trigger, spectrum, levels (Qt-free).

All functions take float arrays in the sound card's range (−1..1) and are
cheap enough to run on ~a second of stereo audio thirty times a second.
"""

from __future__ import annotations

import numpy as np

FLOOR_DB = -90.0
TRIGGER_HYSTERESIS = 0.02


def find_trigger(x: np.ndarray, window_n: int) -> int | None:
    """Index of the last rising zero crossing that still has *window_n* samples after it.

    A crossing only counts once the signal has been below ``-TRIGGER_HYSTERESIS``
    since the previous one, so noise around zero does not re-trigger. None when
    no such crossing exists (silence, DC, or a window as long as the buffer).
    """
    x = np.asarray(x, dtype=np.float32)
    last = len(x) - int(window_n)
    if last < 1:
        return None
    seg = x[:last + 1]
    rising = np.flatnonzero((seg[:-1] <= 0.0) & (seg[1:] > 0.0)) + 1
    if not rising.size:
        return None
    below = np.flatnonzero(x[:last] < -TRIGGER_HYSTERESIS)
    if not below.size:
        return None
    # newest crossing that has a sample under the hysteresis level before it
    # (and after the previous accepted crossing — approximated by "before it")
    for idx in rising[::-1]:
        k = np.searchsorted(below, idx) - 1
        if k >= 0:
            prev = rising[rising < idx]
            if not prev.size or below[k] > prev[-1]:
                return int(idx)
    return None


_windows: dict[int, np.ndarray] = {}


def _hann(n: int) -> np.ndarray:
    w = _windows.get(n)
    if w is None:
        w = _windows[n] = np.hanning(n).astype(np.float64)
    return w


def spectrum_db(x: np.ndarray, sr: int, nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrum of the newest *nfft* samples, in dBFS.

    Hann-windowed and scaled so that a full-scale sine reads 0 dB at its bin.
    Returns ``(freqs_hz, db)`` without the DC bin.
    """
    x = np.asarray(x, dtype=np.float64)
    n = min(int(nfft), len(x))
    if n < 16:
        return np.zeros(0), np.zeros(0)
    w = _hann(n)
    mag = np.abs(np.fft.rfft(x[-n:] * w))
    mag *= 2.0 / w.sum()
    db = 20.0 * np.log10(np.maximum(mag, 10 ** (FLOOR_DB / 20.0)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return freqs[1:], db[1:]


def dominant_hz(freqs: np.ndarray, db: np.ndarray, floor_db: float = -60.0) -> float | None:
    """Frequency of the strongest bin, refined by parabolic interpolation; None if quiet."""
    if not len(db):
        return None
    i = int(np.argmax(db))
    if db[i] < floor_db:
        return None
    if 0 < i < len(db) - 1:
        a, b, c = db[i - 1], db[i], db[i + 1]
        den = a - 2 * b + c
        shift = 0.5 * (a - c) / den if den else 0.0
        step = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
        return float(freqs[i] + shift * step)
    return float(freqs[i])


def phase_between(a: np.ndarray, b: np.ndarray, sr: int, nfft: int = 4096,
                  floor_db: float = -60.0) -> tuple[float, float] | None:
    """``(hz, degrees)``: B's phase relative to A at their strongest common frequency.

    Positive degrees mean B lags A. None when either channel is too quiet there
    to have a meaningful phase.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(int(nfft), len(a), len(b))
    if n < 16:
        return None
    w = _hann(n)
    fa = np.fft.rfft(a[-n:] * w)[1:]
    fb = np.fft.rfft(b[-n:] * w)[1:]
    scale = 2.0 / w.sum()
    ma, mb = np.abs(fa) * scale, np.abs(fb) * scale
    k = int(np.argmax(np.minimum(ma, mb)))
    floor = 10 ** (floor_db / 20.0)
    if ma[k] < floor or mb[k] < floor:
        return None
    deg = float(np.degrees(np.angle(fa[k] * np.conj(fb[k]))))
    hz = (k + 1) * sr / n
    return hz, deg


def levels_db(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rms_db, peak_db)`` per channel of an (n, channels) block, floored at FLOOR_DB."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if not len(x):
        floor = np.full(x.shape[1], FLOOR_DB)
        return floor, floor.copy()
    rms = np.sqrt(np.mean(x * x, axis=0))
    peak = np.max(np.abs(x), axis=0)
    tiny = 10 ** (FLOOR_DB / 20.0)
    return (20.0 * np.log10(np.maximum(rms, tiny)),
            20.0 * np.log10(np.maximum(peak, tiny)))

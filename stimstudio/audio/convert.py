"""Audio → 4 pulse lanes (amp/freq L/R at 40 Hz).

Ported from HWL Viewer's ``hwlfromaudio.py`` (and its duplicate in
``ConversionWorker``), merged into one function with progress and cancel
callbacks. The pitch estimators compute the same values; the per-hop Python
binning loop is replaced by a vectorised reshape (8 hops of 128 samples = one
1/40 s pulse at the 40 960 Hz analysis rate), and the whole file is kept — the
original dropped the last ~0.1 s where no full analysis window fits.

Speed: frames are rows of one strided view over the whole channel, processed
in small cache-friendly blocks (float32 FFTs, which release the GIL) on a
thread pool shared by both channels. Unlike the original 30 s chunks with an
overlap trim — which lost 31 hops at every chunk boundary, so the frequency
lanes drifted ahead of the amplitude — block boundaries cannot misalign frames.
"""

from __future__ import annotations

import os
import threading
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Callable

import numpy as np

from ..core.pulse import FS
from .loader import load_audio

try:
    import scipy.fft as _sfft

    def _rfft(x, n=None):
        return _sfft.rfft(x, n=n, axis=1, workers=1)

    def _irfft(x, n):
        return _sfft.irfft(x, n=n, axis=1, workers=1)

    def _fast_len(n: int) -> int:
        return _sfft.next_fast_len(n, real=True)
except ImportError:  # numpy ≥ 2 keeps float32 precision too
    def _rfft(x, n=None):
        return np.fft.rfft(x, n=n, axis=1)

    def _irfft(x, n):
        return np.fft.irfft(x, n=n, axis=1)

    def _fast_len(n: int) -> int:
        return 1 << (n - 1).bit_length()

SAMPLE_RATE = 40960          # 8 hops of 128 samples per pulse
WINDOW = 4096
HOP = 128
HOPS_PER_PULSE = SAMPLE_RATE // (HOP * FS)
FMIN = 27.5
MAX_FREQ_LOWER_LIMIT = 800.0
SILENCE_DB = -50.0
METHODS = ("rolloff", "centroid", "yin")
BLOCK = 256                  # frames per work item (~0.8 s of audio, a few MB)

Progress = Callable[[str, float], None]


class Cancelled(Exception):
    pass


# ── pitch estimators: one block of frames, shape (m, WINDOW) ────────────────
_SILENCE_RMS = np.float32(10.0 ** (SILENCE_DB / 20.0))


@lru_cache(maxsize=None)
def _hann() -> np.ndarray:
    return np.hanning(WINDOW).astype(np.float32)


def _rms_gate(fr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(fr, dtype=np.float32), axis=1)) > _SILENCE_RMS


def _power(fr: np.ndarray) -> np.ndarray:
    spec = _rfft(fr * _hann())
    return spec.real ** 2 + spec.imag ** 2


@lru_cache(maxsize=None)
def _band(sr: int, fmin: float, fmax: float) -> tuple[slice, np.ndarray]:
    freqs = np.fft.rfftfreq(WINDOW, d=1.0 / sr)
    valid = np.flatnonzero((freqs >= fmin) & (freqs <= fmax))
    return slice(int(valid[0]), int(valid[-1]) + 1), freqs[valid]


def _rolloff_block(fr, sr, fmin, fmax, rolloff=0.5):
    """Frequency below which *rolloff* of the spectral power lies (spectral median)."""
    band, freqs = _band(sr, fmin, fmax)
    cum = np.cumsum(_power(fr)[:, band], axis=1)
    idx = np.argmax(cum >= rolloff * cum[:, -1:], axis=1)
    return np.where(_rms_gate(fr), freqs.astype(np.float32)[idx], 0.0).astype(np.float32)


def _centroid_block(fr, sr, fmin, fmax):
    """Log-frequency spectral centroid."""
    band, freqs = _band(sr, fmin, fmax)
    pv = _power(fr)[:, band].astype(np.float64)
    total = pv.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        lc = np.where(total > 0, (pv @ np.log(freqs)) / total, np.log(fmin))
    return np.where(_rms_gate(fr), np.exp(lc), 0.0).astype(np.float32)


def _yin_block(fr, sr, fmin, fmax, threshold=0.12):
    """Vectorised YIN fundamental-pitch estimate with parabolic interpolation.

    Only lags up to ``sr / fmin`` are searched, so the autocorrelation and the
    cumulative mean are computed for those lags alone (an FFT of
    ``WINDOW + lags`` points is exact for them).
    """
    tau_min = max(1, int(np.ceil(sr / fmax)))
    tau_max = min(WINDOW - 2, int(np.floor(sr / fmin)))
    lags = tau_max + 2
    n_fft = _fast_len(WINDOW + lags)
    spec = _rfft(fr, n=n_fft)
    acf = _irfft(spec.real ** 2 + spec.imag ** 2, n_fft)[:, :lags]
    d = 2.0 * acf[:, :1] - 2.0 * acf
    d[:, 0] = 0.0
    tau_arr = np.arange(1, lags, dtype=np.float32)
    cumsum = np.cumsum(d[:, 1:], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cmndf = np.where(cumsum > 0, d[:, 1:] * tau_arr / cumsum, 1.0)
    search = cmndf[:, tau_min - 1:tau_max]
    n_frames, n_lags = search.shape
    below = search < threshold
    has_below = below.any(axis=1)
    tau_idx = np.where(has_below, np.argmax(below, axis=1), np.argmin(search, axis=1))
    fi = np.arange(n_frames)
    for _ in range(1, 8):
        nxt = np.clip(tau_idx + 1, 0, n_lags - 1)
        lower = search[fi, nxt] < search[fi, tau_idx]
        tau_idx = np.where(has_below & lower & (search[fi, nxt] < threshold), nxt, tau_idx)
    s0 = search[fi, np.clip(tau_idx - 1, 0, n_lags - 1)]
    s1 = search[fi, tau_idx]
    s2 = search[fi, np.clip(tau_idx + 1, 0, n_lags - 1)]
    denom = s0 - 2.0 * s1 + s2
    with np.errstate(invalid="ignore", divide="ignore"):
        interp = np.clip(np.where(np.abs(denom) > 1e-6, 0.5 * (s0 - s2) / denom, 0.0), -1.0, 1.0)
    f0 = (sr / ((tau_idx + tau_min) + interp)).astype(np.float32)
    voiced = _rms_gate(fr) & (s1 < 0.45) & (f0 >= fmin) & (f0 <= fmax)
    return np.where(voiced, f0, 0.0).astype(np.float32)


_ESTIMATORS = {"rolloff": _rolloff_block, "centroid": _centroid_block, "yin": _yin_block}


def frame_rows(y: np.ndarray) -> np.ndarray:
    """``(n_frames, WINDOW)`` strided view (no copy); frame *i* starts at ``i * HOP``."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    if len(y) < WINDOW:
        return np.zeros((0, WINDOW), dtype=np.float32)
    return np.lib.stride_tricks.sliding_window_view(y, WINDOW)[::HOP]


def _whole(method, y, sr, fmin, fmax):
    fr = frame_rows(y)
    if len(fr) == 0:
        return np.zeros(0, np.float32)
    return np.concatenate([_ESTIMATORS[method](fr[b:b + BLOCK], sr, fmin, fmax)
                           for b in range(0, len(fr), BLOCK)])


def yin(y, sr, fmin, fmax):
    return _whole("yin", y, sr, fmin, fmax)


def detect_frequencies(channels: list[np.ndarray], sr: int, method: str, progress=None,
                       cancel=None) -> list[np.ndarray]:
    """Per-hop frequency estimates for each channel, all blocks on one thread pool."""
    fn = _ESTIMATORS[method]
    fmax = sr / 2.0
    rows = [frame_rows(ch) for ch in channels]
    jobs = [(c, b) for c, fr in enumerate(rows) for b in range(0, len(fr), BLOCK)]

    def work(job):
        if cancel and cancel():
            raise Cancelled()
        c, b = job
        return fn(rows[c][b:b + BLOCK], sr, FMIN, fmax)

    parts: list[list[np.ndarray]] = [[] for _ in channels]
    pool = ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 1))
    try:
        futures = [pool.submit(work, j) for j in jobs]
        for k, (job, fut) in enumerate(zip(jobs, futures)):
            parts[job[0]].append(fut.result())
            if progress:
                progress((k + 1) / len(jobs))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return [np.concatenate(p) if p else np.zeros(0, np.float32) for p in parts]


def detect_frequency(y: np.ndarray, sr: int, method: str, progress=None,
                     cancel=None) -> np.ndarray:
    return detect_frequencies([y], sr, method, progress, cancel)[0]


# ── binning and normalisation ───────────────────────────────────────────────
def bin_amplitude(ch: np.ndarray, n: int) -> np.ndarray:
    """RMS of each 1/40 s block (the last block may be short)."""
    per = HOP * HOPS_PER_PULSE
    sq = np.zeros(n * per, dtype=np.float64)
    sq[:len(ch)] = ch.astype(np.float64) ** 2
    counts = np.full(n, per, dtype=np.float64)
    counts[-1] = len(ch) - per * (n - 1)
    return np.sqrt(sq.reshape(n, per).sum(axis=1) / np.maximum(counts, 1))


def bin_frequency(f0: np.ndarray, n: int) -> np.ndarray:
    """Median of the non-zero hop estimates in each pulse; 0 where there are none."""
    hops = np.zeros(n * HOPS_PER_PULSE, dtype=np.float64)
    m = min(len(f0), len(hops))
    hops[:m] = f0[:m]
    block = hops.reshape(n, HOPS_PER_PULSE)
    masked = np.where(block > 0, block, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows → 0 below
        med = np.nanmedian(masked, axis=1)
    return np.nan_to_num(med, nan=0.0)


def fill_missing(freq: np.ndarray, default: float = 400.0) -> np.ndarray:
    """Hold the last valid frequency through gaps; leading gaps take the first one."""
    valid = freq != 0.0
    if not valid.any():
        return np.full_like(freq, default)
    idx = np.where(valid, np.arange(len(freq)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = freq[idx]
    out[:int(np.argmax(valid))] = freq[int(np.argmax(valid))]
    return out


def normalisation_maximum(left: np.ndarray, right: np.ndarray) -> float:
    """A maximum that ignores up to 2 % outliers (ported ``choose_normalisation_maximum``)."""
    def pmax(p):
        return max(np.percentile(left, p), np.percentile(right, p))
    known_good = pmax(98.0)
    percentile = 100.0
    while percentile > 98.0:
        value = pmax(percentile)
        if value <= known_good * 1.1:
            return float(value)
        percentile -= 0.5
    return float(known_good * 1.1)


# ── cached file conversion ──────────────────────────────────────────────────
_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_lock = threading.Lock()
_CACHE_MAX = 8


def _key(path, method) -> tuple:
    st = os.stat(path)
    return (os.path.abspath(path), st.st_size, int(st.st_mtime_ns), method)


def audio_to_pulses(path, method: str = "rolloff", progress: Progress | None = None,
                    cancel: Callable[[], bool] | None = None) -> np.ndarray:
    """Convert an audio file to an (N, 4) pulse array (N = duration × 40, rounded up).

    Results are cached per file (path + size + mtime) and method.
    """
    if method not in _ESTIMATORS:
        raise ValueError(f"unknown frequency method {method!r}")

    def report(stage, frac):
        if progress:
            progress(stage, frac)
        if cancel and cancel():
            raise Cancelled()

    report("Loading audio", 0.0)
    key = _key(path, method)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit.copy()
    data, sr = load_audio(path, SAMPLE_RATE)
    left = data[:, 0]
    right = data[:, 1] if data.shape[1] > 1 else left
    n = max(1, int(np.ceil(len(left) / (HOP * HOPS_PER_PULSE))))

    mono = right is left or np.array_equal(left, right)
    label = "Detecting frequencies" + (" (mono)" if mono else " (left + right)")
    report(label, 0.0)
    f0 = detect_frequencies([left] if mono else [left, right], sr, method,
                            lambda f: report(label, f), cancel)
    f0_l, f0_r = f0[0], f0[0] if mono else f0[1]

    report("Normalising", 1.0)
    amp_l, amp_r = bin_amplitude(left, n), bin_amplitude(right, n)
    freq_l = fill_missing(bin_frequency(np.where(f0_l > sr / 2, 0, f0_l), n))
    freq_r = freq_l if mono else fill_missing(bin_frequency(np.where(f0_r > sr / 2, 0, f0_r), n))

    max_amp = normalisation_maximum(amp_l, amp_r)
    max_freq = max(normalisation_maximum(freq_l, freq_r), MAX_FREQ_LOWER_LIMIT)
    out = np.zeros((n, 4), dtype=np.float32)
    if max_amp > 0:
        out[:, 0], out[:, 1] = np.clip(amp_l / max_amp, 0, 1), np.clip(amp_r / max_amp, 0, 1)
    out[:, 2], out[:, 3] = np.clip(freq_l / max_freq, 0, 1), np.clip(freq_r / max_freq, 0, 1)
    with _lock:
        _cache[key] = out.copy()
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return out

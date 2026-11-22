"""Modulation signals from an audio file, at the 40 Hz control rate.

Signals (all 0..1):

* ``envelope`` — loudness (RMS with fast attack / slower release);
* ``onsets``   — a decaying pulse at every detected beat / transient;
* ``low`` / ``mid`` / ``high`` — energy below 200 Hz, 200–2000 Hz, above 2 kHz.

Results are cached per file (path + size + mtime), so the background renderer
and the GUI share one analysis.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Callable

import numpy as np

from ..core.pulse import FS
from .loader import load_audio

SIGNALS = ("envelope", "onsets", "low", "mid", "high")
SIGNAL_LABELS = {"envelope": "Envelope", "onsets": "Beats / onsets", "low": "Low band",
                 "mid": "Mid band", "high": "High band"}
ANALYSIS_SR = 22050
N_FFT = 2048
BANDS = {"low": (20.0, 200.0), "mid": (200.0, 2000.0), "high": (2000.0, 11025.0)}
ONSET_DECAY_S = 0.15


def _normalise(x: np.ndarray, pct: float = 99.0) -> np.ndarray:
    top = np.percentile(x, pct) if len(x) else 0.0
    return np.clip(x / top, 0.0, 1.0) if top > 0 else np.zeros_like(x)


def _follower(x: np.ndarray, attack_s: float, release_s: float) -> np.ndarray:
    a_up = 1.0 - np.exp(-1.0 / (attack_s * FS)) if attack_s > 0 else 1.0
    a_dn = 1.0 - np.exp(-1.0 / (release_s * FS)) if release_s > 0 else 1.0
    out = np.empty_like(x)
    y = 0.0
    for i, v in enumerate(x):
        y += (v - y) * (a_up if v > y else a_dn)
        out[i] = y
    return out


def _pick_onsets(flux: np.ndarray) -> np.ndarray:
    """Indices of transient peaks: local maxima clearly above the local median.

    A median threshold (±0.5 s) is robust: one huge transient does not mask its
    neighbours the way a mean / standard-deviation threshold would.
    """
    if len(flux) < 3:
        return np.zeros(0, dtype=int)
    w = FS // 2
    pad = np.pad(flux, w, mode="edge")
    local_med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1), axis=1)
    thresh = local_med * 1.5 + 0.1 * np.percentile(flux, 95)
    local_max = np.r_[False, (flux[1:-1] >= flux[:-2]) & (flux[1:-1] > flux[2:]), False]
    peaks = np.flatnonzero(local_max & (flux > thresh))
    # at most one onset per 80 ms
    keep, last = [], -10 ** 9
    for p in peaks:
        if p - last >= round(0.08 * FS):
            keep.append(p)
            last = p
    return np.array(keep, dtype=int)


def analyze_samples(mono: np.ndarray, sr: int, progress=None,
                    cancel: Callable[[], bool] | None = None) -> dict[str, np.ndarray]:
    n = max(1, int(np.ceil(len(mono) / sr * FS)))
    starts = np.round(np.arange(n) * sr / FS).astype(np.int64) - N_FFT // 2
    padded = np.pad(mono.astype(np.float32), (N_FFT, N_FFT))
    window = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    band_idx = {k: (freqs >= lo) & (freqs < hi) for k, (lo, hi) in BANDS.items()}
    rms = np.zeros(n)
    bands = {k: np.zeros(n) for k in BANDS}
    flux = np.zeros(n)
    prev = None
    hop = int(round(sr / FS))
    block = 1024
    for b0 in range(0, n, block):
        if cancel and cancel():
            from .convert import Cancelled
            raise Cancelled()
        idx = starts[b0:b0 + block, None] + N_FFT + np.arange(N_FFT)[None, :]
        fr = padded[idx]                                   # (m, N_FFT)
        centre = fr[:, N_FFT // 2 - hop: N_FFT // 2 + hop]
        rms[b0:b0 + len(fr)] = np.sqrt(np.mean(centre.astype(np.float64) ** 2, axis=1))
        mag = np.abs(np.fft.rfft(fr * window, axis=1))
        power = mag ** 2
        for k, sel in band_idx.items():
            bands[k][b0:b0 + len(fr)] = np.sqrt(power[:, sel].sum(axis=1))
        logmag = np.log1p(mag * 10.0)
        if prev is None:
            prev = logmag[:1]
        diff = np.diff(np.vstack([prev, logmag]), axis=0)
        flux[b0:b0 + len(fr)] = np.maximum(diff, 0.0).sum(axis=1)
        prev = logmag[-1:]
        if progress:
            progress(min(1.0, (b0 + block) / n))

    env = _follower(_normalise(rms), 0.01, 0.2)
    onsets = np.zeros(n)
    decay = np.exp(-np.arange(int(ONSET_DECAY_S * FS * 5)) / (ONSET_DECAY_S * FS))
    for p in _pick_onsets(flux):
        seg = onsets[p:p + len(decay)]
        np.maximum(seg, decay[:len(seg)], out=seg)
    out = {"envelope": env, "onsets": onsets}
    for k in BANDS:
        out[k] = _follower(_normalise(bands[k]), 0.01, 0.12)
    return {k: v.astype(np.float32) for k, v in out.items()}


# ── cached file analysis ────────────────────────────────────────────────────
_cache: "OrderedDict[tuple, dict[str, np.ndarray]]" = OrderedDict()
_lock = threading.Lock()
_CACHE_MAX = 16


def _key(path) -> tuple:
    st = os.stat(path)
    return (os.path.abspath(path), st.st_size, int(st.st_mtime_ns))


def cached_analysis(path) -> dict[str, np.ndarray] | None:
    """The cached analysis of *path*, or None (also None if the file is missing)."""
    try:
        key = _key(path)
    except OSError:
        return None
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
        return hit


def analyze_file(path, progress=None, cancel=None) -> dict[str, np.ndarray]:
    """Analyse *path* (cached). Raises OSError if the file is missing."""
    hit = cached_analysis(path)
    if hit is not None:
        return hit
    key = _key(path)
    data, sr = load_audio(path, ANALYSIS_SR)
    signals = analyze_samples(data.mean(axis=1), sr, progress, cancel)
    for arr in signals.values():
        arr.setflags(write=False)
    with _lock:
        _cache[key] = signals
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return signals


def silent_signals(n: int) -> dict[str, np.ndarray]:
    return {k: np.zeros(n, dtype=np.float32) for k in SIGNALS}

"""Phase-coded three-phase stereo audio → pulse lanes (A/B or triphase).

A three-phase file carries one carrier in both channels; the position lives in
the channel levels and their phase difference: ``L = Re(a·e^{iωt})``,
``R = Re(b·e^{iωt})`` with phasors ``a``, ``b`` (see ``core.triphase``). The
analytic signal of each channel (Hilbert transform, here an FFT per chunk) is
``z ≈ phasor · e^{iωt}``, so per 1/40 s pulse

* ``mean |z_L|²``, ``mean |z_R|²`` → |a|, |b| (RMS of the envelope);
* ``arg mean(z_L · conj z_R)`` → the phase of a against b (b is taken real);
* ``|mean(z_L · conj z_R)| / sqrt(…)`` → coherence: near 1 for a real
  three-phase file, low for plain stereo with unrelated channels;
* ``arg Σ z[n]·conj z[n−1]`` (both channels) → the carrier frequency.

Files come in two kinds: an **unbroken carrier** (restim-style) and **bursts**
of carrier repeated at a pulse rate (like the wavelet output). Bursts are
found on a ~2 kHz envelope: there the amplitude is the burst peak (an RMS
would fall with the duty cycle) and the burst rate can become the frequency.

:func:`analyse_file` does the heavy pass (cached per file); :func:`to_pulses`
turns the result into an ``(N, 4)`` array for any choice of options, cheaply.
"""

from __future__ import annotations

import functools
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core import triphase as tp
from ..core.pulse import FS
from ..outputs.base import FrequencyMap
from .convert import Cancelled, fill_missing, normalisation_maximum
from .loader import load_audio

__all__ = ["Cancelled", "TriAudioAnalysis", "analyse_file", "to_pulses", "MODES", "OUTPUTS",
           "FREQ_SOURCES", "default_hz_range"]

MODES = ("auto", "continuous", "pulsed")
OUTPUTS = ("channels", "electrodes", "triphase")
FREQ_SOURCES = ("carrier", "burst", "constant")

FFT_LEN = 1 << 19            # samples per analytic-signal FFT (chunk + padding)
PAD = 4096                   # context on each side of a chunk (Hilbert edge effects)
ENV_RATE = 2000              # Hz, envelope used to find bursts
SILENCE_DB = -60.0           # dBFS: quieter pulses carry no frequency / phase
BURST_ON = 0.35              # envelope above this share of the local peak = inside a burst
BURST_WINDOW_S = 0.25        # burst rate = edges counted in this window around each pulse
PULSED_DUTY = 0.7            # less of the active time inside bursts than this → pulsed
FREQ_MEDIAN = 5              # pulses in the frequency median filter
FIT_MARGIN = 0.1             # auto range: lowest point → 0.1, highest → 0.9
MIN_FIT_SPAN_HZ = 1.0        # … but never narrower than this

_SILENCE_AMP = 10.0 ** (SILENCE_DB / 20.0)

Progress = Callable[[str, float], None]


@dataclass(frozen=True)
class TriAudioAnalysis:
    """Per-pulse results of :func:`analyse_file` (arrays are read-only, shared)."""

    sr: int
    duration_s: float
    a: np.ndarray            # complex, RMS level and phase against b
    b: np.ndarray            # complex (real ≥ 0)
    peak_a: np.ndarray       # burst-peak levels of L and R
    peak_b: np.ndarray
    carrier_hz: np.ndarray   # 0 where silent
    burst_hz: np.ndarray     # 0 where no bursts
    coherence: np.ndarray    # 0..1, 0 where silent
    duty: float              # share of the active time inside bursts (1 = unbroken)
    detected_mode: str       # "continuous" | "pulsed"

    @property
    def n(self) -> int:
        return len(self.a)

    @property
    def active(self) -> np.ndarray:
        return np.maximum(np.abs(self.a), np.abs(self.b)) > _SILENCE_AMP

    def _median(self, arr: np.ndarray) -> float:
        v = arr[arr > 0]
        return float(np.median(v)) if len(v) else 0.0

    @property
    def carrier_median(self) -> float:
        return self._median(self.carrier_hz)

    @property
    def burst_median(self) -> float:
        return self._median(self.burst_hz)

    @property
    def mean_coherence(self) -> float:
        """Level-weighted coherence over the file (1 = clean three-phase)."""
        w = np.abs(self.a) * np.abs(self.b)
        return float((self.coherence * w).sum() / w.sum()) if w.sum() > 0 else 0.0


def default_hz_range(freq_source: str) -> tuple[float, float]:
    """Carrier: the continuous-audio carrier range; bursts / constant: Howl's pulse range."""
    if freq_source == "carrier":
        return 500.0, 1000.0
    fm = FrequencyMap()
    return float(fm.min_hz), float(fm.max_hz)


# ── analysis ────────────────────────────────────────────────────────────────
try:
    import scipy.fft as _sfft

    def _rfft(x, n):
        return _sfft.rfft(x, n=n, workers=-1)

    def _ifft(x):
        return _sfft.ifft(x, workers=-1)
except ImportError:
    _rfft, _ifft = np.fft.rfft, np.fft.ifft


def _analytic(x: np.ndarray) -> np.ndarray:
    """Analytic signal of a real 1-D block (zero-padded FFT to FFT_LEN)."""
    n = len(x)
    nf = max(FFT_LEN, 1 << max(0, n - 1).bit_length())
    spec = _rfft(x, nf)
    full = np.zeros(nf, dtype=spec.dtype)
    full[0] = spec[0]
    full[1:nf // 2] = 2.0 * spec[1:nf // 2]
    full[nf // 2] = spec[nf // 2]
    return _ifft(full)[:n].astype(np.complex64)


def _moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) == 0:
        return x
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    lo = np.clip(np.arange(len(x)) - n // 2, 0, len(x))
    hi = np.clip(lo + n, 0, len(x))
    return ((c[hi] - c[lo]) / np.maximum(hi - lo, 1)).astype(np.float32)


def _demodulate(left, right, sr, edges, progress, cancel):
    """Per-pulse sums over the whole file, chunk by chunk (pulse-aligned chunks)."""
    n = len(edges) - 1
    per = sr / FS
    chunk = max(1, int((FFT_LEN - 2 * PAD) / per) - 1)
    sums = {k: np.zeros(n, np.float64) for k in ("paa", "pbb")}
    cross = np.zeros(n, np.complex128)
    lag = np.zeros(n, np.complex128)
    peak_a = np.zeros(n, np.float32)
    peak_b = np.zeros(n, np.float32)
    hop = max(1, sr // ENV_RATE)
    env = np.zeros(len(left) // hop + 2, np.float64)
    smooth = max(1, sr // 2000)                      # ~0.5 ms against envelope ripple
    total = len(left)
    for p0 in range(0, n, chunk):
        if cancel and cancel():
            raise Cancelled()
        p1 = min(n, p0 + chunk)
        s0, s1 = int(edges[p0]), int(edges[p1])
        w0, w1 = max(0, s0 - PAD), min(total, s1 + PAD)
        zl = _analytic(left[w0:w1])[s0 - w0:s1 - w0]
        zr = _analytic(right[w0:w1])[s0 - w0:s1 - w0]
        local = (edges[p0:p1] - s0).astype(np.int64)
        al, ar = np.abs(zl), np.abs(zr)
        pl, pr = al.astype(np.float64) ** 2, ar.astype(np.float64) ** 2
        sums["paa"][p0:p1] = np.add.reduceat(pl, local)
        sums["pbb"][p0:p1] = np.add.reduceat(pr, local)
        cross[p0:p1] = np.add.reduceat((zl * np.conj(zr)).astype(np.complex128), local)
        lg = np.zeros(len(zl), np.complex128)
        lg[1:] = zl[1:] * np.conj(zl[:-1]) + zr[1:] * np.conj(zr[:-1])
        lag[p0:p1] = np.add.reduceat(lg, local)
        peak_a[p0:p1] = np.maximum.reduceat(_moving_mean(al, smooth), local)
        peak_b[p0:p1] = np.maximum.reduceat(_moving_mean(ar, smooth), local)
        bins = (np.arange(s0, s1) // hop) - s0 // hop
        env[s0 // hop:s0 // hop + int(bins[-1]) + 1] += np.bincount(bins, weights=pl + pr)
        if progress:
            progress(p1 / n)
    counts = np.diff(edges).astype(np.float64)
    n_env = (total + hop - 1) // hop
    env_counts = np.full(n_env, hop, np.float64)
    env_counts[-1] = total - hop * (n_env - 1)
    env = np.sqrt(env[:n_env] / env_counts)
    return sums["paa"] / counts, sums["pbb"] / counts, cross / counts, lag, peak_a, peak_b, \
        env.astype(np.float32), sr / hop


def _bursts(env: np.ndarray, env_sr: float, n: int):
    """Burst rate per pulse (0 = none) and the duty cycle over the active time."""
    block = max(1, int(env_sr * 0.05))
    nb = (len(env) + block - 1) // block
    padded = np.zeros(nb * block, np.float32)
    padded[:len(env)] = env
    bmax = padded.reshape(nb, block).max(axis=1)
    near = np.maximum(bmax, np.maximum(np.r_[bmax[1:], 0], np.r_[0, bmax[:-1]]))
    local_peak = np.repeat(near, block)[:len(env)]
    active = local_peak > _SILENCE_AMP
    on = (env > BURST_ON * local_peak) & active
    duty = float(on[active].mean()) if active.any() else 1.0
    rising = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    te = rising / env_sr
    centres = (np.arange(n) + 0.5) / FS
    lo = np.searchsorted(te, centres - BURST_WINDOW_S / 2)
    hi = np.searchsorted(te, centres + BURST_WINDOW_S / 2)
    cnt = hi - lo
    rate = np.zeros(n, np.float64)
    ok = cnt >= 3
    if ok.any() and len(te):
        span = te[np.minimum(hi[ok] - 1, len(te) - 1)] - te[np.minimum(lo[ok], len(te) - 1)]
        rate[ok] = np.where(span > 0, (cnt[ok] - 1) / np.maximum(span, 1e-9), 0.0)
    return rate, duty


def analyse(data: np.ndarray, sr: int, progress: Progress | None = None,
            cancel: Callable[[], bool] | None = None) -> TriAudioAnalysis:
    """Analyse ``(samples, channels)`` audio (at least two channels: L, R)."""
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("a three-phase file needs two channels (stereo)")
    left = np.ascontiguousarray(data[:, 0], np.float32)
    right = np.ascontiguousarray(data[:, 1], np.float32)
    total = len(left)
    if total == 0:
        raise ValueError("the file has no audio")
    per = sr / FS
    n = max(1, int(np.ceil(total / per)))
    edges = np.minimum(np.round(np.arange(n + 1) * per).astype(np.int64), total)
    edges[-1] = total
    n = int(np.count_nonzero(np.diff(edges) > 0))           # drop a zero-length tail pulse
    edges = edges[:n + 1]

    def report(frac):
        if progress:
            progress("Demodulating", frac)

    paa, pbb, cross, lag, peak_a, peak_b, env, env_sr = _demodulate(
        left, right, sr, edges, report, cancel)
    if progress:
        progress("Finding bursts", 1.0)
    amp_a, amp_b = np.sqrt(paa), np.sqrt(pbb)
    active = np.maximum(amp_a, amp_b) > _SILENCE_AMP
    phase = np.angle(cross)
    a = (amp_a * np.exp(1j * phase)).astype(np.complex64)
    b = amp_b.astype(np.complex64)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = np.where(active, np.abs(cross) / np.sqrt(paa * pbb), 0.0)
    coh = np.nan_to_num(np.clip(coh, 0.0, 1.0)).astype(np.float32)
    carrier = np.where(active, np.angle(lag) * sr / (2 * np.pi), 0.0)
    carrier = np.clip(carrier, 0.0, None).astype(np.float32)
    rate, duty = _bursts(env, env_sr, n)      # kept between bursts (pulses there are silent)
    rate = rate.astype(np.float32)
    has_rate = (rate > 0)[active]
    pulsed = duty < PULSED_DUTY and has_rate.size > 0 and has_rate.mean() > 0.5
    arrays = [a, b, peak_a, peak_b, carrier, rate, coh]
    for arr in arrays:
        arr.setflags(write=False)
    return TriAudioAnalysis(sr=sr, duration_s=total / sr, a=a, b=b, peak_a=peak_a,
                            peak_b=peak_b, carrier_hz=carrier, burst_hz=rate, coherence=coh,
                            duty=duty, detected_mode="pulsed" if pulsed else "continuous")


_cache: "OrderedDict[tuple, TriAudioAnalysis]" = OrderedDict()
_lock = threading.Lock()
_CACHE_MAX = 4


def analyse_file(path, progress: Progress | None = None,
                 cancel: Callable[[], bool] | None = None) -> TriAudioAnalysis:
    """Load and analyse a file at its own sample rate (cached per path + size + mtime)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, int(st.st_mtime_ns))
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    if progress:
        progress("Loading audio", 0.0)
    data, sr = load_audio(path)
    if cancel and cancel():
        raise Cancelled()
    result = analyse(data, sr, progress, cancel)
    with _lock:
        _cache[key] = result
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return result


# ── options → pulses ────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=32)
def _electrode_full(cal: tp.Calibration) -> float:
    """Largest electrode current over the disk at full volume (sensation scale)."""
    g = np.linspace(0.0, 2 * np.pi, 721)
    xs = np.concatenate([np.cos(g), 0.5 * np.cos(g), [0.0]])
    ys = np.concatenate([np.sin(g), 0.5 * np.sin(g), [0.0]])
    a, b = tp.output_phasors(1.0, xs, ys, cal)
    return float(max(np.abs(2 * a - b).max(), np.abs(2 * b - a).max(), 1e-9))


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) == 0:
        return x
    k = np.ones(n) / n
    pad = n // 2
    xp = np.concatenate([np.repeat(x[:1], pad), x, np.repeat(x[-1:], n - 1 - pad)])
    if np.iscomplexobj(x):
        return np.convolve(xp.real, k, "valid") + 1j * np.convolve(xp.imag, k, "valid")
    return np.convolve(xp, k, "valid")


def _median_filter(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or len(x) < n:
        return x
    pad = n // 2
    xp = np.concatenate([np.repeat(x[:1], pad), x, np.repeat(x[-1:], pad)])
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, n), axis=1)


def _hold_bursts(an: TriAudioAnalysis) -> tuple[np.ndarray, np.ndarray]:
    """Burst-peak phasors, each pulse taking the strongest burst within one burst
    period around it — below 40 bursts a second some pulses fall between bursts."""
    level = an.peak_a.astype(np.float64) ** 2 + an.peak_b.astype(np.float64) ** 2
    rate = an.burst_hz.astype(np.float64)
    width = np.where(rate > 0, np.ceil(FS / np.maximum(rate, 1e-9)), 1).astype(np.int64)
    width = np.clip(width | 1, 1, 2 * FS + 1)                   # odd: centred windows
    pick = np.arange(an.n)
    for w in np.unique(width):
        if w <= 1:
            continue
        half = w // 2
        padded = np.concatenate([np.full(half, -1.0), level, np.full(half, -1.0)])
        best = np.argmax(np.lib.stride_tricks.sliding_window_view(padded, w), axis=1)
        sel = width == w
        pick[sel] = (np.arange(an.n) + best - half)[sel]
    pick = np.clip(pick, 0, an.n - 1)
    a = an.peak_a[pick].astype(np.float64) * np.exp(1j * np.angle(an.a[pick]))
    return a, an.peak_b[pick].astype(np.complex128)


def freq_curve(an: TriAudioAnalysis, freq_source: str, default: float) -> np.ndarray:
    """Hz per pulse as the lanes get it: carrier or burst rate, gaps held, median
    filtered; *default* where the file has none at all."""
    raw = (an.carrier_hz if freq_source == "carrier" else an.burst_hz).astype(np.float64)
    return _median_filter(fill_missing(raw, default), FREQ_MEDIAN)


def fitted_hz_range(an: TriAudioAnalysis, freq_source: str,
                    margin: float = FIT_MARGIN) -> tuple[float, float] | None:
    """The 0..1 range that puts the lowest point of the frequency curve at
    *margin* and the highest at ``1 − margin`` (0.1 .. 0.9 by default).

    A curve flatter than ``MIN_FIT_SPAN_HZ`` is widened to that around its
    middle. None if the file has no such frequency.
    """
    raw = an.carrier_hz if freq_source == "carrier" else an.burst_hz
    if freq_source == "constant" or not np.any(raw > 0):
        return None
    hz = freq_curve(an, freq_source, 0.0)
    lo, hi = float(hz.min()), float(hz.max())
    if hi - lo < MIN_FIT_SPAN_HZ:
        mid = (lo + hi) / 2
        lo, hi = mid - MIN_FIT_SPAN_HZ / 2, mid + MIN_FIT_SPAN_HZ / 2
    pad = (hi - lo) * margin / (1.0 - 2.0 * margin)
    return max(0.0, lo - pad), hi + pad


def _scale(values: list[np.ndarray], normalize: bool, full: float) -> float:
    if normalize:
        m = normalisation_maximum(values[0], values[-1])
        return m if m > 0 else 1.0
    return full


def to_pulses(an: TriAudioAnalysis, *, mode: str = "auto", output: str = "channels",
              freq_source: str = "carrier", hz_range: tuple[float, float] | None = None,
              const_hz: float | None = None, normalize: bool = True, smooth_s: float = 0.0,
              cal: tp.Calibration = tp.DEFAULT_CAL) -> np.ndarray:
    """``(N, 4)`` float32: ``amp A, amp B, freq A, freq B`` (outputs "channels" /
    "electrodes") or ``volume, x, y, freq`` ("triphase"), all 0..1."""
    if mode not in MODES or output not in OUTPUTS or freq_source not in FREQ_SOURCES:
        raise ValueError(f"bad options: {mode!r}, {output!r}, {freq_source!r}")
    if mode == "auto":
        mode = an.detected_mode
    if mode == "pulsed":
        a, b = _hold_bursts(an)
    else:
        a, b = an.a.astype(np.complex128), an.b.astype(np.complex128)
    n_smooth = int(round(max(0.0, smooth_s) * FS))
    a, b = _smooth(a, n_smooth), _smooth(b, n_smooth)

    out = np.zeros((an.n, 4), np.float32)
    if output == "triphase":
        vol, x, y = tp.phasors_to_tri(a, b, cal)
        vol = vol / _scale([vol], normalize, 1.0)
        out[:, tp.VOL] = np.clip(vol, 0.0, 1.0)
        out[:, tp.X], out[:, tp.Y] = tp.xy_to_unit(x), tp.xy_to_unit(y)
    else:
        if output == "electrodes":
            amp_a, amp_b = np.abs(2 * a - b), np.abs(2 * b - a)
            full = _electrode_full(cal)
        else:
            amp_a, amp_b = np.abs(a), np.abs(b)
            full = 1.0
        k = _scale([amp_a, amp_b], normalize, full)
        out[:, 0], out[:, 1] = np.clip(amp_a / k, 0, 1), np.clip(amp_b / k, 0, 1)

    lo, hi = hz_range if hz_range is not None else default_hz_range(freq_source)
    if freq_source == "constant":
        hz = np.full(an.n, (lo + hi) / 2 if const_hz is None else const_hz, np.float64)
    else:
        hz = freq_curve(an, freq_source, (lo + hi) / 2)
    norm = np.clip((hz - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    if output == "triphase":
        out[:, tp.FREQ] = norm
    else:
        out[:, 2] = out[:, 3] = norm
    return out

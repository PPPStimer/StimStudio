"""Is an audio file three-phase or two-channel? A quick look at a few excerpts.

Three excerpts of 30 s (one at a random place in each third of the file; a
short file is taken whole) are demodulated with ``triphase_decode.analyse``.
A three-phase file is **one carrier in both channels** (high phase coherence)
whose **phase difference is away from 0° and 180°** — that difference is what
steers the current to the third electrode. Plain stereo has unrelated channels
(low coherence), or the same signal in phase / in antiphase (Δ ≈ 0° / 180°),
which a two-channel conversion represents fully.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .convert import Cancelled
from .loader import load_audio
from .triphase_decode import analyse

SEGMENTS = 3
SEGMENT_S = 30.0
COHERENT = 0.85              # one carrier in both channels from here
OFF_AXIS_DEG = 15.0          # phase difference this far from 0° / 180° steers
OFF_AXIS_SHARE = 0.2         # … for at least this share of the (level-weighted) time
SILENT_CHANNEL_DB = -30.0    # one channel this much quieter than the other: "silent"

Progress = Callable[[str, float], None]


@dataclass
class ProbeResult:
    kind: str                         # "triphase" | "stereo"
    channels: int
    sr: int
    duration_s: float
    segments: list[tuple[float, float]] = field(default_factory=list)   # (start, length) s
    coherence: float = 0.0            # level-weighted, 0..1
    off_axis: float = 0.0             # share of the time with Δ away from 0° / 180°
    phase_deg: float = 0.0            # typical |Δ|, 0..180
    phase_span: tuple[float, float] = (0.0, 0.0)   # 10th..90th percentile of |Δ|
    balance_db: float = 0.0           # level L − R
    carrier_hz: float = 0.0
    carrier_span: tuple[float, float] = (0.0, 0.0)  # 10th..90th percentile
    signal: str = "continuous"        # "continuous" | "pulsed"
    burst_hz: float = 0.0
    silent: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def is_triphase(self) -> bool:
        return self.kind == "triphase"


def pick_segments(duration_s: float, n: int = SEGMENTS, length_s: float = SEGMENT_S,
                  rng: random.Random | None = None) -> list[tuple[float, float]]:
    """``(start, length)`` excerpts: one at a random place in each of *n* equal
    parts of the file; the whole file when it is not longer than ``n · length``."""
    if duration_s <= n * length_s:
        return [(0.0, max(0.0, duration_s))]
    rng = rng or random.Random()
    part = duration_s / n
    return [(part * i + rng.uniform(0.0, part - length_s), length_s) for i in range(n)]


def _read(path, segments, cancel) -> tuple[list[np.ndarray], int]:
    """Read just the excerpts (soundfile seeks); otherwise load the file and slice."""
    try:
        import soundfile as sf
        out = []
        with sf.SoundFile(str(path)) as f:
            sr = f.samplerate
            for start, length in segments:
                if cancel and cancel():
                    raise Cancelled()
                f.seek(int(start * sr))
                out.append(f.read(int(round(length * sr)), dtype="float32", always_2d=True))
        return out, int(sr)
    except Cancelled:
        raise
    except Exception:  # noqa: BLE001 — not seekable by libsndfile: decode it all
        data, sr = load_audio(path)
        return [data[int(s * sr):int((s + n) * sr)] for s, n in segments], sr


def file_info(path) -> tuple[float, int]:
    """``(duration_s, channels)`` without decoding the file (when possible)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.frames / info.samplerate, info.channels
    except Exception:  # noqa: BLE001
        data, sr = load_audio(path)
        return len(data) / sr, data.shape[1]


def _wpct(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][min(len(cw) - 1, int(np.searchsorted(cw, q * cw[-1])))])


def probe(excerpts: list[np.ndarray], sr: int, duration_s: float,
          segments: list[tuple[float, float]], progress: Progress | None = None,
          cancel: Callable[[], bool] | None = None) -> ProbeResult:
    channels = excerpts[0].shape[1] if excerpts and excerpts[0].ndim == 2 else 1
    res = ProbeResult(kind="stereo", channels=channels, sr=sr, duration_s=duration_s,
                      segments=list(segments))
    if channels < 2:
        res.reasons.append("Mono file: one channel cannot carry a phase difference.")
        return res
    parts = []
    for i, ex in enumerate(excerpts):
        if cancel and cancel():
            raise Cancelled()
        if len(ex):
            parts.append(analyse(ex, sr, cancel=cancel))
        if progress:
            progress("Analysing excerpts", (i + 1) / len(excerpts))

    def cat(name):
        return np.concatenate([getattr(p, name) for p in parts]) if parts else np.zeros(0)

    a, b, coh = cat("a"), cat("b"), cat("coherence")
    amp_a, amp_b = np.abs(a), np.abs(b)
    w = amp_a * amp_b
    if not parts or w.sum() <= 0 and max(amp_a.max(initial=0), amp_b.max(initial=0)) <= 1e-3:
        res.silent = True
        res.reasons.append("The excerpts are silent: nothing to judge by.")
        return res
    res.balance_db = 20 * math.log10(max(float(np.sqrt(np.mean(amp_a ** 2))), 1e-9)
                                     / max(float(np.sqrt(np.mean(amp_b ** 2))), 1e-9))
    res.coherence = float((coh * w).sum() / w.sum()) if w.sum() > 0 else 0.0
    delta = np.degrees(np.abs(np.angle(a)))                       # b is real: Δ = arg a
    wc = w * coh
    if wc.sum() > 0:
        off = (delta > OFF_AXIS_DEG) & (delta < 180.0 - OFF_AXIS_DEG)
        res.off_axis = float(wc[off].sum() / wc.sum())
        res.phase_deg = _wpct(delta, wc, 0.5)
        res.phase_span = (_wpct(delta, wc, 0.1), _wpct(delta, wc, 0.9))
    carrier = cat("carrier_hz")
    carrier = carrier[carrier > 0]
    if len(carrier):
        res.carrier_hz = float(np.median(carrier))
        res.carrier_span = (float(np.percentile(carrier, 10)), float(np.percentile(carrier, 90)))
    pulsed = sum(p.detected_mode == "pulsed" for p in parts)
    res.signal = "pulsed" if pulsed * 2 > len(parts) else "continuous"
    bursts = cat("burst_hz")
    res.burst_hz = float(np.median(bursts[bursts > 0])) if np.any(bursts > 0) else 0.0
    _judge(res)
    return res


def _judge(res: ProbeResult) -> None:
    r = res.reasons
    quiet = abs(res.balance_db) >= -SILENT_CHANNEL_DB
    if quiet:
        side = "right" if res.balance_db > 0 else "left"
        r.append(f"The {side} channel is (nearly) silent ({abs(res.balance_db):.0f} dB quieter).")
    if res.coherence >= COHERENT and not quiet:
        r.append(f"Both channels carry one signal (phase coherence {res.coherence:.2f}).")
        lo, hi = res.phase_span
        if res.off_axis >= OFF_AXIS_SHARE:
            res.kind = "triphase"
            span = (f"steady at ≈ {res.phase_deg:.0f}°" if hi - lo < 10 else
                    f"moving between {lo:.0f}° and {hi:.0f}°")
            r.append(f"Their phase difference is {span} — away from 0° / 180° "
                     f"{res.off_axis * 100:.0f} % of the time: it steers the current to the "
                     "third electrode.")
        else:
            how = "in phase (Δ ≈ 0°)" if res.phase_deg < 90 else "in antiphase (Δ ≈ 180°)"
            r.append(f"…but {how} {(1 - res.off_axis) * 100:.0f} % of the time: the same "
                     "signal on both sides, nothing for a third electrode.")
    elif not quiet:
        r.append(f"The channels carry different signals (phase coherence "
                 f"{res.coherence:.2f}; a three-phase file is ≥ {COHERENT:.2f}).")
    if res.carrier_hz and res.coherence >= COHERENT and not quiet:
        lo, hi = res.carrier_span
        if hi - lo < 0.05 * res.carrier_hz:
            r.append(f"Carrier ≈ {res.carrier_hz:.0f} Hz, steady.")
        elif hi - lo < 0.5 * res.carrier_hz:
            r.append(f"Carrier ≈ {res.carrier_hz:.0f} Hz, gliding {lo:.0f}–{hi:.0f} Hz.")
        else:
            r.append(f"No single carrier: the pitch wanders {lo:.0f}–{hi:.0f} Hz "
                     "(music or noise rather than a stim tone).")
    if res.signal == "pulsed" and res.burst_hz:
        r.append(f"Carrier in bursts, ≈ {res.burst_hz:.1f} per second.")


def probe_file(path, progress: Progress | None = None, cancel: Callable[[], bool] | None = None,
               rng: random.Random | None = None, n: int = SEGMENTS,
               length_s: float = SEGMENT_S) -> ProbeResult:
    """Read *n* excerpts of *length_s* and judge the file (see the module doc)."""
    if progress:
        progress("Reading excerpts", 0.0)
    duration, _channels = file_info(path)
    segments = pick_segments(duration, n, length_s, rng)
    excerpts, sr = _read(path, segments, cancel)
    return probe(excerpts, sr, duration, segments, progress, cancel)

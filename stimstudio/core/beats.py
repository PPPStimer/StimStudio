"""Beat marks from signals: peak picking, quantising.

Pure numpy helpers behind "Beat marks from peaks…" and the pattern editor's
Quantize. Signals are 0..1 at ``FS`` pulses per second.
"""

from __future__ import annotations

import numpy as np

from .model import Clip, Mark
from .pulse import FS

PEAK_METHODS = ("cross", "max")
PEAK_LABELS = {"cross": "Level crossing (attack)", "max": "Maxima above level"}


def level_signal(data: np.ndarray, domain: str = "pulse") -> np.ndarray:
    """The loudness a peak search looks at: max(L, R) amplitude, or the
    triphase volume."""
    if len(data) == 0:
        return np.zeros(0, dtype=np.float64)
    if domain == "triphase":
        return np.asarray(data[:, 0], dtype=np.float64)
    return np.maximum(data[:, 0], data[:, 1]).astype(np.float64)


def clip_level_signal(clip: Clip, cache=None, bpm: float = 120.0,
                      domain: str = "pulse") -> np.ndarray:
    from .render import render_clip
    return level_signal(render_clip(clip, cache, bpm, domain), domain)


def detect_peaks(signal, level: float, method: str = "cross", min_gap_s: float = 0.1,
                 hysteresis: float = 0.05, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """Beat positions in *signal* as ``(pulse indices, heights)``.

    ``cross``: a beat where the signal rises to *level*; the next one only
    after it fell below ``level − hysteresis``. The height is the maximum
    reached before that fall. Beats closer than *min_gap_s* to the previous
    one are dropped.

    ``max``: local maxima at or above *level*; of maxima closer than
    *min_gap_s* the strongest stays.
    """
    s = np.asarray(signal, dtype=np.float64)
    gap = max(0, int(round(min_gap_s * fs)))
    if len(s) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    if method == "cross":
        return _crossings(s, level, gap, max(0.0, hysteresis))
    if method == "max":
        return _maxima(s, level, gap)
    raise ValueError(f"unknown peak method: {method!r}")


def _crossings(s, level, gap, hyst):
    idx, heights = [], []
    armed, last, height = True, None, 0.0
    rearm = level - hyst
    for i, v in enumerate(s):
        if armed:
            if v >= level:
                armed = False
                if last is None or i - last >= gap:
                    idx.append(i)
                    heights.append(v)
                    last = i
                    height = v
                else:
                    height = None          # too close: this rise is ignored
        else:
            if height is not None and v > height:
                height = heights[-1] = v
            if v < rearm:
                armed = True
    return np.array(idx, dtype=np.int64), np.array(heights, dtype=np.float64)


def _maxima(s, level, gap):
    n = len(s)
    left = np.concatenate([[-np.inf], s[:-1]])
    right = np.concatenate([s[1:], [-np.inf]])
    cand = np.flatnonzero((s >= level) & (s > left) & (s >= right))
    if gap > 1 and len(cand) > 1:
        keep: list[int] = []
        taken = np.zeros(n, dtype=bool)
        for i in cand[np.argsort(-s[cand], kind="stable")]:
            lo, hi = max(0, i - gap + 1), min(n, i + gap)
            if not taken[lo:hi].any():
                keep.append(int(i))
                taken[i] = True
        cand = np.array(sorted(keep), dtype=np.int64)
    return cand.astype(np.int64), s[cand]


def marks_from_peaks(idx, heights, fs: int = FS) -> list[Mark]:
    return [Mark(float(i) / fs, float(min(1.0, max(0.0, h)))) for i, h in zip(idx, heights)]


def quantize_marks(marks: list[Mark], grid_s: float, strength: float = 1.0,
                   offset_s: float = 0.0) -> list[Mark]:
    """Pull marks towards the grid (*strength* 0..1). Mark times are source
    times; *offset_s* is the timeline time of source time 0 (``clip.start_s −
    clip.trim_s``), so the grid is the project's. Marks that land on the same
    pulse merge (the louder one stays)."""
    if grid_s <= 0:
        return [m.clone() for m in marks]
    k = min(1.0, max(0.0, float(strength)))
    out: dict[int, Mark] = {}
    for m in marks:
        t = m.t_s + offset_s
        q = round(t / grid_s) * grid_s
        nm = m.clone()
        nm.t_s = max(0.0, t + (q - t) * k - offset_s)
        slot = round(nm.t_s * FS)
        if slot not in out or nm.velocity > out[slot].velocity:
            out[slot] = nm
    return sorted(out.values(), key=lambda m: m.t_s)

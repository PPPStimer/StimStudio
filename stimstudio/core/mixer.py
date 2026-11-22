"""Combine several pulse layers (clips on a track, tracks on the master) into one.

Amplitude and frequency cannot be mixed the same way:

* **Amplitude** follows the layer's blend mode — ``add`` (sum, clamped),
  ``max``, ``multiply`` (gate / modulate what is below) or ``replace``
  (this layer overrides everything below it where it is active).
* **Frequency** is the amplitude-weighted mean of the active layers, so the
  loudest layer dominates the pitch. A tiny weight floor keeps the frequency
  defined (and a single layer passes through unchanged) when amplitude is 0.
  Where no layer is active at all the last known frequency is held.

A layer can leave either part out per side (lane mutes): without its
amplitude it still steers the frequency, weighted by the amplitude it would
have had; without its frequency it only adds amplitude. That way one track can
supply the amplitude and another the frequency.

Layers are applied in order; later layers sit "above" earlier ones.
"""

from __future__ import annotations

import numpy as np

from .pulse import AMP_COLS, FREQ_COLS, N_COLS

#: Weight floor for frequency averaging (see module docstring).
FREQ_WEIGHT_EPS = 1e-6
#: Frequency used before the first active pulse.
DEFAULT_FREQ = 0.5

_AMP = list(AMP_COLS)
_FREQ = list(FREQ_COLS)


class Accumulator:
    """Accumulates layers over an ``n``-pulse timeline."""

    def __init__(self, n: int):
        self.n = int(n)
        self.amp = np.zeros((self.n, 2), dtype=np.float64)
        self.num = np.zeros((self.n, 2), dtype=np.float64)
        self.den = np.zeros((self.n, 2), dtype=np.float64)
        self.active = np.zeros(self.n, dtype=bool)

    def add(self, data: np.ndarray, start: int = 0, blend: str = "add",
            gain: float = 1.0, active: np.ndarray | None = None,
            freq_weight: np.ndarray | None = None, amp_on=(True, True),
            freq_on=(True, True)) -> None:
        """Mix *data* (an (m, 4) array) in at pulse *start*.

        *active* optionally marks which rows of *data* carry signal; by default
        all of them do. Rows outside the timeline are dropped. *freq_weight*
        ((m, 2), default: the amplitude columns) weights the layer's frequency.
        *amp_on* / *freq_on* (per side A, B) False leave that part out entirely.
        """
        m = len(data)
        lo, hi = max(0, start), min(self.n, start + m)
        if hi <= lo:
            return
        src = data[lo - start:hi - start]
        sl = slice(lo, hi)
        mask = np.ones(hi - lo, dtype=bool) if active is None else active[lo - start:hi - start]
        if not mask.any():
            return
        am = mask[:, None] & np.asarray(amp_on, dtype=bool)[None, :]
        fm = mask[:, None] & np.asarray(freq_on, dtype=bool)[None, :]

        la = src[:, _AMP].astype(np.float64) * float(gain)
        lf = src[:, _FREQ].astype(np.float64)
        fw = la if freq_weight is None else (
            freq_weight[lo - start:hi - start].astype(np.float64) * float(gain))
        w = fw + FREQ_WEIGHT_EPS

        amp, num, den = self.amp[sl], self.num[sl], self.den[sl]
        if blend == "add":
            amp += np.where(am, la, 0.0)
            num += np.where(fm, w * lf, 0.0)
            den += np.where(fm, w, 0.0)
        elif blend == "max":
            np.copyto(amp, np.maximum(amp, la), where=am)
            num += np.where(fm, w * lf, 0.0)
            den += np.where(fm, w, 0.0)
        elif blend == "multiply":
            np.copyto(amp, amp * la, where=am)
        elif blend == "replace":
            np.copyto(amp, la, where=am)
            np.copyto(num, w * lf, where=fm)
            np.copyto(den, w, where=fm)
        else:
            raise ValueError(f"unknown blend mode: {blend!r}")
        self.active[sl] |= mask

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(pulses (n, 4) float32, active (n,) bool)``."""
        out = np.zeros((self.n, N_COLS), dtype=np.float32)
        out[:, _AMP] = np.clip(self.amp, 0.0, 1.0)
        for side, col in enumerate(_FREQ):
            out[:, col] = hold_frequency(self.num[:, side], self.den[:, side])
        return out, self.active.copy()


def hold_frequency(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` where defined; elsewhere hold the previous defined value."""
    n = len(num)
    valid = den > 0
    if not valid.any():
        return np.full(n, DEFAULT_FREQ, dtype=np.float32)
    vals = np.zeros(n, dtype=np.float64)
    vals[valid] = num[valid] / den[valid]
    idx = np.where(valid, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(valid))
    out = vals[idx]
    out[:first] = vals[first]
    return np.clip(out, 0.0, 1.0).astype(np.float32)

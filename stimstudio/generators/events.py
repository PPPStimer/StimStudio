"""Small helpers shared by the event-based generators (bursts, taps, touches, beats)."""

from __future__ import annotations

import numpy as np

from ..core.pulse import FS


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def place_max(arr: np.ndarray, start: int, seg: np.ndarray) -> None:
    """``arr[start:start+len(seg)] = max(arr, seg)``, clipped to the array (start may be < 0)."""
    n = len(arr)
    a, b = max(0, start), min(n, start + len(seg))
    if a < b:
        np.maximum(arr[a:b], seg[a - start:b - start], out=arr[a:b])


def hold_last(values: np.ndarray, valid: np.ndarray, default: float) -> np.ndarray:
    """Forward-fill *values* from the positions where *valid* is True."""
    idx = np.where(valid, np.arange(len(values)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, values[np.maximum(idx, 0)], default)


def iso_compensation(freq, amount: float, curve: float = 0.7):
    """Amplitude factor that keeps the felt intensity roughly constant over pulse frequency.

    Perceived electrotactile intensity grows with pulse rate, so the level has to
    drop as frequency rises (e.g. Dideriksen 2015, JNER). 1 at freq 0, ``1 - amount``
    at freq 1, with a concave (power-law) curve in between.
    """
    return 1.0 - float(amount) * np.clip(freq, 0.0, 1.0) ** float(curve)


def ms_to_pulses(ms: float, minimum: int = 1) -> int:
    return max(minimum, int(round(float(ms) / 1000.0 * FS)))

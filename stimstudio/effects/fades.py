"""Amplitude fades and edge crossfades on pulse arrays (amplitude only for fades)."""

from __future__ import annotations

import numpy as np

from ..core.pulse import AMP_COLS, FREQ_COLS, FS


def _cols(channels, amp=True, freq=True) -> list[int]:
    chs = set(channels)
    sides = [side for side, on in enumerate(("A" in chs, "B" in chs)) if on]
    cols = []
    if amp:
        cols += [AMP_COLS[s] for s in sides]
    if freq:
        cols += [FREQ_COLS[s] for s in sides]
    return cols


def _ramp_len(n: int, seconds: float, fs: int) -> int:
    if seconds <= 0 or n == 0:
        return 0
    return min(n, max(1, round(seconds * fs)))


def fade_in(pulses: np.ndarray, seconds: float, channels=("A", "B"), fs: int = FS) -> np.ndarray:
    """Ramp amplitude up from exactly 0 over the first *seconds*. In place; returns *pulses*."""
    k = _ramp_len(len(pulses), seconds, fs)
    if k:
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
        pulses[:k, _cols(channels, freq=False)] *= ramp[:, None]
    return pulses


def fade_out(pulses: np.ndarray, seconds: float, channels=("A", "B"), fs: int = FS) -> np.ndarray:
    """Ramp amplitude down to exactly 0 at the last pulse. In place; returns *pulses*."""
    n = len(pulses)
    k = _ramp_len(n, seconds, fs)
    if k:
        ramp = np.linspace(1.0, 0.0, k, dtype=np.float32)
        pulses[n - k:, _cols(channels, freq=False)] *= ramp[:, None]
    return pulses


def crossfade_edges(pulses: np.ndarray, start: int, end: int, seconds: float,
                    channels=("A", "B"), fs: int = FS) -> np.ndarray:
    """Blend both edges of [start, end) into the samples just outside it.

    Ported from HWL Viewer. At the first sample the region equals its left
    neighbour, at the last it equals its right neighbour, easing back to its
    own values across the window. Returns the new slice.
    """
    seg = pulses[start:end].copy()
    n = len(seg)
    cols = _cols(channels)
    w = max(1, int(round(seconds * fs)))
    if n == 0 or not cols:
        return seg

    half = max(1, n // 2)
    k_left = min(w, half) if start > 0 else 0
    k_right = min(w, half) if end < len(pulses) else 0

    if k_left:
        t = (np.arange(k_left, dtype=np.float32) / max(1, k_left - 1))[:, None]
        anchor = pulses[start - 1][cols]
        seg[:k_left, cols] = anchor * (1.0 - t) + seg[:k_left, cols] * t
    if k_right:
        t = (np.arange(k_right, dtype=np.float32) / max(1, k_right - 1))[:, None]
        anchor = pulses[end][cols]
        seg[n - k_right:, cols] = seg[n - k_right:, cols] * (1.0 - t) + anchor * t
    return seg

"""Deterministic smooth-noise helpers used by activities and channel motion."""

from __future__ import annotations

import numpy as np

from ..core.pulse import FS


def smooth_random(n: int, speed: float, rng, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Smoothly interpolated random curve in [lo, hi].

    *speed* ~ random control points per second.
    """
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    pts = max(2, int(round(speed * n / FS)) + 1)
    base = rng.uniform(0.0, 1.0, size=pts)
    curve = np.interp(np.linspace(0.0, pts - 1, n), np.arange(pts), base)
    return (lo + (hi - lo) * curve).astype(np.float32)


def stepped_random(n: int, hold_s: float, rng, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Random value held for *hold_s* seconds, then jumping to a new value."""
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    hold = max(1, int(round(hold_s * FS)))
    blocks = int(np.ceil(n / hold))
    vals = rng.uniform(lo, hi, size=blocks)
    return np.repeat(vals, hold)[:n].astype(np.float32)


def fractal_noise(n: int, base_speed: float, rng, octaves: int = 3) -> np.ndarray:
    """Sum of smooth-random octaves, normalized to [0, 1]. Simplex-like rolling."""
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    amp = 1.0
    total = 0.0
    speed = base_speed
    for _ in range(max(1, octaves)):
        out += amp * smooth_random(n, speed, rng)
        total += amp
        amp *= 0.5
        speed *= 2.0
    return (out / total).astype(np.float32)


# -- generator ------------------------------------------------------------
from .base import Generator, ParamSpec, register  # noqa: E402
from ..core.pulse import PulseTrack  # noqa: E402

NOISE_KINDS = ("smooth", "stepped", "fractal")


@register
class NoiseGenerator(Generator):
    """Random amplitude/frequency curves, independent per channel."""

    type_id = "noise"
    label = "Noise"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("kind", "Kind", "choice", "smooth", choices=NOISE_KINDS),
            ParamSpec("speed", "Speed (pts/s)", "float", 1.0, 0.05, 20.0, 0.05),
            ParamSpec("amp_min", "Amp min", "float", 0.3, 0.0, 1.0, 0.01),
            ParamSpec("amp_max", "Amp max", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("freq_min", "Freq min", "float", 0.2, 0.0, 1.0, 0.01),
            ParamSpec("freq_max", "Freq max", "float", 0.8, 0.0, 1.0, 0.01),
            ParamSpec("link", "Same on A/B", "bool", False),
        ]

    def render(self, n: int, params: dict, seed: int = 0) -> PulseTrack:
        p = self.merged_params(params)
        rng = np.random.default_rng(seed)
        speed = float(p["speed"])
        kind = p["kind"]

        def curve(lo, hi):
            if kind == "stepped":
                return stepped_random(n, 1.0 / max(speed, 1e-3), rng, lo, hi)
            if kind == "fractal":
                return (lo + (hi - lo) * fractal_noise(n, speed, rng)).astype(np.float32)
            return smooth_random(n, speed, rng, lo, hi)

        amp_lo, amp_hi = float(p["amp_min"]), float(p["amp_max"])
        f_lo, f_hi = float(p["freq_min"]), float(p["freq_max"])
        l_amp, l_freq = curve(amp_lo, amp_hi), curve(f_lo, f_hi)
        if p["link"]:
            r_amp, r_freq = l_amp.copy(), l_freq.copy()
        else:
            r_amp, r_freq = curve(amp_lo, amp_hi), curve(f_lo, f_hi)
        return PulseTrack(l_amp, r_amp, l_freq, r_freq)

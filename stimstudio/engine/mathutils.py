"""Interpolation, helper math, and a seeded RNG context.

Ported from Howl's Utils.kt. All activity randomness flows through the RNG set
by :func:`set_rng`, so a given seed reproduces a given render. The RNG is
thread-local: renders running in parallel worker threads never share state.
"""

from __future__ import annotations

import math
import random
import threading
from typing import Sequence

# --- seeded RNG context (one per thread) ---------------------------------
_local = threading.local()


def set_rng(seed) -> None:
    """Reset this thread's RNG. Call once at the start of a render."""
    _local.rng = random.Random(seed)


def rng() -> random.Random:
    r = getattr(_local, "rng", None)
    if r is None:
        r = _local.rng = random.Random()
    return r


def next_double(a: float = 0.0, b: float = 1.0) -> float:
    """Equivalent to Kotlin Random.nextDouble(a, b): uniform in [a, b)."""
    return a + (b - a) * rng().random()


def next_bool() -> bool:
    return rng().random() < 0.5


def next_int(n: int) -> int:
    """Kotlin Random.nextInt(n): 0..n-1."""
    return rng().randrange(n)


def random_in_range(lo: float, hi: float, bias: float = 1.0) -> float:
    """Ported randomInRange: biased random in [lo, hi].

    bias > 1 favors lower values, bias < 1 favors higher values, bias == 1 uniform.
    """
    assert bias > 0
    mn, mx = (lo, hi) if lo <= hi else (hi, lo)
    if mn == mx:
        return mn
    u = rng().random()
    skewed = u ** bias
    return mn + skewed * (mx - mn)


def random_int_in_range(lo: int, hi: int) -> int:
    """Kotlin IntRange.random(): inclusive of both ends."""
    return rng().randint(lo, hi)


def choice(seq: Sequence):
    """Kotlin List.random() / Array.random(): uniform element."""
    return seq[rng().randrange(len(seq))]


# --- scalar helpers ------------------------------------------------------
def lerp(start: float, end: float, fraction: float) -> float:
    return start + (end - start) * fraction


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def scale_between(value: float, a: float, b: float) -> float:
    """Double.scaleBetween: map value in [0,1] to [a,b] and clamp."""
    result = a + (b - a) * value
    lo, hi = (a, b) if a <= b else (b, a)
    return clamp(result, lo, hi)


def scale_velocity(velocity: float, sensitivity: float) -> float:
    if velocity == 0.0:
        return 0.0
    av = abs(velocity)
    return av / (av + sensitivity)


def smoothstep(t: float) -> float:
    c = clamp(t, 0.0, 1.0)
    return c * c * (3.0 - 2.0 * c)


def calculate_feel_adjustment(value: float, feel: float) -> float:
    return clamp(value ** (1.0 / feel), 0.0, 1.0)


def calculate_positional_effect(amplitude: float, position: float, strength: float):
    """Equal-power pan: returns (ampA, ampB)."""
    effective = 0.5 * (1 - strength) + position * strength
    amp_a = amplitude * math.sqrt(max(0.0, 1 - effective))
    amp_b = amplitude * math.sqrt(max(0.0, effective))
    return amp_a, amp_b


def calculate_engulf_effect(amplitude, position, channel_a_point, channel_b_point):
    def channel_amp(pos, engulf):
        distance = abs(pos - engulf)
        falloff = 0.8
        if pos <= engulf:
            return 1.0 if engulf == 0.0 else math.sqrt(pos / engulf)
        return math.sqrt(max(0.0, 1.0 - distance * falloff))

    return channel_amp(position, channel_a_point) * amplitude, \
        channel_amp(position, channel_b_point) * amplitude


# --- interpolation -------------------------------------------------------
def linear_interpolate(t, t0, p0, t1, p1) -> float:
    if t0 >= t1:
        return p0
    h = (t - t0) / (t1 - t0)
    return p0 + h * (p1 - p0)


def linear_interpolate_with_velocity(t, t0, p0, t1, p1):
    if t0 >= t1:
        return p0, 0.0
    h = (t - t0) / (t1 - t0)
    position = p0 + h * (p1 - p0)
    velocity = (p1 - p0) / (t1 - t0)
    return position, velocity


def _hermite_pos_and_factors(t, t0, p0, m0, t1, p1, m1):
    if t0 >= t1:
        return p0, 0.0, 0.0, 0.0
    h = (t - t0) / (t1 - t0)
    h_sq = h * h
    h_cu = h_sq * h
    position = (
        p0 * (2 * h_cu - 3 * h_sq + 1)
        + m0 * (h_cu - 2 * h_sq + h) * (t1 - t0)
        + p1 * (-2 * h_cu + 3 * h_sq)
        + m1 * (h_cu - h_sq) * (t1 - t0)
    )
    return position, h, h_sq, h_cu


def hermite_interpolate(t, t0, p0, m0, t1, p1, m1) -> float:
    return _hermite_pos_and_factors(t, t0, p0, m0, t1, p1, m1)[0]


def hermite_interpolate_with_velocity(t, t0, p0, m0, t1, p1, m1):
    position, h, h_sq, _ = _hermite_pos_and_factors(t, t0, p0, m0, t1, p1, m1)
    if t1 == t0:
        return position, 0.0
    dpdh = (
        (6 * h_sq - 6 * h) * p0
        + (3 * h_sq - 4 * h + 1) * m0 * (t1 - t0)
        + (-6 * h_sq + 6 * h) * p1
        + (3 * h_sq - 2 * h) * m1 * (t1 - t0)
    )
    return position, dpdh / (t1 - t0)

"""How clips on one track meet: overlaps and butt joints.

A track's ``overlap`` mode decides what happens where clips overlap:

* ``crossfade`` — clips are layered like images with transparency: a clip that
  starts later lies *on top*. Where it overlaps the clip below, its opacity
  ramps up (S-curve) across the whole overlap, so one clip hands over to the
  next and the levels always sum to 1 — no bump, no dip. A clip lying inside a
  longer one replaces it, entering and leaving with ramps ``join_s`` long. A
  manual fade on an edge that lies inside another clip is ignored there: the
  crossfade takes its place.
* ``mix`` — the clips add up (clamped at 1); the original behaviour, kept for
  projects made before overlap modes existed.
* ``max`` — the louder clip wins at every moment.

Independently, ``smooth_joins`` rounds off butt joints — one clip ending where
the next begins — over ``join_s``: amplitude, frequency (and the triphase
position) glide from the end of one clip to the start of the next instead of
jumping.

Everything here works in pulses on plain spans so the renderer and the song
editor (which draws the crossfades) share one definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OVERLAP_MODES = ("crossfade", "mix", "max")
OVERLAP_LABELS = {"crossfade": "Crossfade", "mix": "Mix (add)", "max": "Max (louder wins)"}
DEFAULT_JOIN_S = 0.25
JOIN_CHOICES_S = (0.1, 0.25, 0.5, 1.0, 2.0)
#: clips whose edges are at most this many pulses apart count as a butt joint
JOIN_TOLERANCE = 1


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def ramp_up(k: int) -> np.ndarray:
    """*k* samples of an S-curve from (just above) 0 to (just below) 1."""
    return smoothstep((np.arange(k, dtype=np.float64) + 0.5) / max(1, k))


@dataclass
class Layer:
    """How one clip sits among the others on its track (all in pulses)."""

    start: int
    n: int
    order: int = 0           # stacking order: higher lies on top
    ramp_in: int = 0         # opacity ramp at the start (crossfade from what is below)
    ramp_out: int = 0        # opacity ramp at the end (back to what is below)
    covered_in: bool = False     # the start lies inside another clip
    covered_out: bool = False    # the end lies inside another clip

    @property
    def end(self) -> int:
        return self.start + self.n


def layout_layers(spans, join_n: int) -> list[Layer]:
    """Stacking and crossfade ramps for clips given as ``(start, n)`` spans in
    track order. Clips that start later lie on top; equal starts keep the
    track order. Returns the layers in the input order."""
    layers = [Layer(int(s), max(0, int(n))) for s, n in spans]
    stack = sorted(range(len(layers)), key=lambda i: (layers[i].start, i))
    for rank, i in enumerate(stack):
        layers[i].order = rank
    for i, a in enumerate(layers):
        if a.n == 0:
            continue
        below = [b for b in layers if b is not a and b.n and b.order < a.order]
        under_start = [b for b in below if b.start <= a.start < b.end]
        if under_start:
            top = max(under_start, key=lambda b: b.order)
            a.ramp_in = top.end - a.start if top.end < a.end else join_n
        if any(b.end > a.end for b in under_start):
            a.ramp_out = join_n
        if a.ramp_in + a.ramp_out > a.n:          # short clip: share its length
            total = a.ramp_in + a.ramp_out
            a.ramp_in = a.ramp_in * a.n // total
            a.ramp_out = a.n - a.ramp_in
        a.covered_in = bool(under_start)
        a.covered_out = any(b is not a and b.n and b.start < a.end < b.end for b in layers)
    return layers


def presence(layer: Layer) -> np.ndarray:
    """Opacity of a clip over its own span."""
    p = np.ones(layer.n, dtype=np.float64)
    if layer.ramp_in:
        p[:layer.ramp_in] = ramp_up(layer.ramp_in)
    if layer.ramp_out:
        p[layer.n - layer.ramp_out:] = ramp_up(layer.ramp_out)[::-1]
    return p


def composite_gains(layers: list[Layer]) -> list[np.ndarray]:
    """Per-clip gain curves for ``crossfade`` mode ("over" compositing): a clip's
    opacity times the transparency of every clip above it. Wherever clips
    overlap the gains add up to 1."""
    pres = [presence(L) for L in layers]
    gains = [p.copy() for p in pres]
    for i, a in enumerate(layers):
        for j, b in enumerate(layers):
            if b.order <= a.order or b.n == 0:
                continue
            lo, hi = max(a.start, b.start), min(a.end, b.end)
            if hi <= lo:
                continue
            gains[i][lo - a.start:hi - a.start] *= 1.0 - pres[j][lo - b.start:hi - b.start]
    return gains


def butt_joints(spans, tolerance: int = JOIN_TOLERANCE) -> list[int]:
    """Pulse positions where one clip ends and another begins (no overlap, no gap)."""
    edges = sorted((int(s), int(s) + int(n)) for s, n in spans if n > 0)
    out = []
    for k, (s, _e) in enumerate(edges):
        ends = [e for s2, e in edges[:k] if s2 < s]
        if not ends:
            continue
        last = max(ends)
        # nothing may cover the joint: the previous clips all end at it
        if abs(last - s) <= tolerance and not any(s2 < s and e > s + tolerance
                                                  for s2, e in edges[:k]):
            out.append(s)
    return sorted(set(out))


def smooth_joints(arr: np.ndarray, joints, half_n: int, cols, spans=None) -> None:
    """Round off the step at every joint in place: inside ``[j - h, j + h)`` the
    left clip's values (held after the joint) blend into the right clip's
    (held before it) along an S-curve. *h* is limited to half of the shorter
    neighbour when *spans* are given."""
    n = len(arr)
    cols = list(cols)
    for j in joints:
        h = int(half_n)
        if spans is not None:
            left = [sn for s, sn in spans if s + sn == j or abs(s + sn - j) <= JOIN_TOLERANCE]
            right = [sn for s, sn in spans if abs(s - j) <= JOIN_TOLERANCE]
            if left and right:
                h = min(h, max(left) // 2, max(right) // 2)
        lo, hi = max(0, j - h), min(n, j + h)
        if h < 1 or not 0 < j < n or j <= lo or j >= hi:
            continue
        seg = arr[lo:hi][:, cols].astype(np.float64)
        k = j - lo
        a = seg.copy()
        a[k:] = arr[j - 1, cols]          # left clip held past the joint
        b = seg.copy()
        b[:k] = arr[j, cols]              # right clip held before the joint
        s = smoothstep((np.arange(hi - lo) + 0.5) / (hi - lo))[:, None]
        arr[lo:hi, cols] = (a * (1.0 - s) + b * s).astype(arr.dtype)


def audible_clips(clips) -> list:
    """The clips that take part in a track's mix, in mixing order (by start)."""
    from .model import GeneratorSource, PatternSource, PulseDataSource
    return sorted((c for c in clips if not c.muted and c.length_s > 0
                   and isinstance(c.source, (GeneratorSource, PulseDataSource, PatternSource))),
                  key=lambda c: c.start_s)


def clip_spans(clips, offset: int = 0) -> list[tuple[int, int]]:
    from .pulse import n_pulses, pulse_index
    return [(pulse_index(c.start_s) - offset, n_pulses(c.length_s)) for c in clips]


def track_layers(track) -> dict[str, Layer]:
    """``clip id → Layer`` for a ``crossfade`` track ({} for the other modes)."""
    if track.overlap != "crossfade" or track.type not in ("standard", "triphase"):
        return {}
    from .pulse import n_pulses
    clips = audible_clips(track.clips)
    layers = layout_layers(clip_spans(clips), n_pulses(track.join_s))
    return {c.id: L for c, L in zip(clips, layers)}

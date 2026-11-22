"""Pattern clips: one clip of a pool played at every beat mark.

    marks ─► resolve (pin · weighted random · sequence) ─► one entry per mark
          ─► placements (entry render fitted to the gap) ─► mix_clips

Every pool entry is rendered once at its own length (the source render is
cached like any clip's); a placement is a slice / loop / time-stretch of that
render, so hundreds of marks cost no more generator simulation than one.

Determinism: the random pick draws exactly one number per mark, whether the
mark is pinned or not, so pinning or unpinning one mark never changes the
picks of the others.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from .model import (PATTERN_ANCHORS, Clip, GeneratorSource, Mark, PatternSource, PoolEntry,
                    PulseDataSource)
from .pulse import FS, n_pulses, pulse_index

#: sequence tokens that mean "nothing plays at this mark"
RESTS = ("-", ".")
FIT_LABELS = {"cut": "Own length, cut at neighbours", "own": "Own length (may overlap)",
              "loop": "Loop to fill", "stretch": "Stretch to fill"}
ANCHOR_LABELS = {"start": "Clip start on the mark", "center": "Clip middle on the mark",
                 "end": "Clip end on the mark"}
MODE_LABELS = {"random": "Random (weighted)", "sequence": "Sequence"}


def key_name(i: int) -> str:
    """A, B, …, Z, AA, AB, …"""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def next_key(pool) -> str:
    used = {e.key for e in pool}
    i = 0
    while key_name(i) in used:
        i += 1
    return key_name(i)


def parse_sequence(text: str) -> list[str]:
    """Pool keys (and rests) separated by spaces or commas, upper-cased."""
    return [t.upper() for t in re.split(r"[\s,;]+", text.strip()) if t]


def sequence_errors(text: str, pool) -> list[str]:
    """Tokens of *text* that name no pool entry (rests are fine)."""
    keys = {e.key for e in pool}
    return [t for t in parse_sequence(text) if t not in keys and t not in RESTS]


def playable(entry: PoolEntry) -> bool:
    c = entry.clip
    return (not c.muted and c.length_s > 0
            and isinstance(c.source, (GeneratorSource, PulseDataSource)))


def resolve(pattern: PatternSource) -> list[tuple[Mark, PoolEntry | None]]:
    """The entry that plays at every mark (``None``: a rest), marks by time."""
    marks = sorted(pattern.marks, key=lambda m: m.t_s)
    pool = [e for e in pattern.pool if playable(e)]
    by_key = {e.key: e for e in pool}
    rng = np.random.default_rng(int(pattern.seed))
    draws = rng.random(len(marks))
    weights = np.array([max(0.0, float(e.weight)) for e in pool], dtype=np.float64)
    cum = np.cumsum(weights)
    seq = parse_sequence(pattern.sequence) if pattern.mode == "sequence" else []
    out = []
    for i, mark in enumerate(marks):
        entry = by_key.get(mark.pin) if mark.pin else None
        if entry is None:
            if pattern.mode == "sequence":
                entry = by_key.get(seq[i % len(seq)]) if seq else None
            elif len(pool) and cum[-1] > 0:
                k = int(np.searchsorted(cum, draws[i] * cum[-1], side="right"))
                entry = pool[min(k, len(pool) - 1)]
        out.append((mark, entry))
    return out


def velocity_gain(velocity: float, depth: float) -> float:
    v = min(1.0, max(0.0, float(velocity)))
    d = min(1.0, max(0.0, float(depth)))
    return 1.0 - d + d * v


def _stretch(data: np.ndarray, m: int) -> np.ndarray:
    n = len(data)
    if n == m or n == 0:
        return data
    x = np.linspace(0.0, n - 1, m)
    return np.stack([np.interp(x, np.arange(n), data[:, c]) for c in range(data.shape[1])],
                    1).astype(np.float32)


def _fit(data: np.ndarray, fit: str, mark: int, zone: tuple[int, int], anchor: float,
         join_n: int, amp_cols) -> tuple[int, np.ndarray]:
    """``(start pulse, data)`` of one clip placed with its anchor point on
    *mark*. *zone* is the stretch of time the mark owns (see ``placements``):
    ``cut`` trims what sticks out of it (fading the cut edges), ``loop`` and
    ``stretch`` fill it, ``own`` ignores it."""
    n = len(data)
    lo, hi = zone
    if fit == "own":
        return mark - round(anchor * n), data
    if fit == "cut":
        start = mark - round(anchor * n)
        a, b = max(start, lo), min(start + n, hi)
        if b <= a:
            return a, data[:0]
        out = np.array(data[a - start:b - start], dtype=np.float32)
        m = len(out)
        k = min(m, join_n)
        if a > start and k:                             # head cut off: fade in
            out[:k, amp_cols] *= np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        if b < start + n and k:                         # tail cut off: fade out
            out[m - k:, amp_cols] *= np.linspace(1.0, 0.0, k, dtype=np.float32)[:, None]
        return a, out
    m = hi - lo
    if m <= 0 or n == 0:
        return lo, data[:0]
    if fit == "loop":
        # the clip's anchor point lands on the mark; repeats run both ways
        phase = round(anchor * n) - (mark - lo)
        return lo, data[(np.arange(m) + phase) % n]
    if fit == "stretch":
        return lo, _stretch(data, m)
    raise ValueError(f"unknown pattern fit: {fit!r}")


class _PulseSkippingCache:
    """Cache proxy for the placement mix: placements are throw-away pulse
    slices, caching them would only evict real source renders."""

    def __init__(self, cache):
        self.cache = cache

    def get(self, key):
        return None if key.startswith("pulses:") else self.cache.get(key)

    def put(self, key, arr):
        if not key.startswith("pulses:"):
            self.cache.put(key, arr)


def placements(pattern: PatternSource, total_n: int, cache=None, bpm: float = 120.0,
               domain: str = "pulse") -> list[Clip]:
    """The clips the pattern plays over its first *total_n* pulses.

    Each mark owns a zone around it: with the anchor at fraction *a* of the
    clip (start 0, middle ½, end 1) the zone runs from ``a`` of the way back
    to the previous mark (the block start for the first) to ``1 − a`` of the
    way on to the next one (the block end for the last). Zones of
    neighbouring marks meet exactly, so ``loop`` / ``stretch`` tile the block
    and ``cut`` never lets two clips overlap.
    """
    from .render import render_clip
    plan = [(pulse_index(m.t_s), m, e) for m, e in resolve(pattern)]
    renders: dict[str, np.ndarray] = {}
    join_n = n_pulses(pattern.join_s)
    amp_cols = [0] if domain == "triphase" else [0, 1]   # triphase: volume only
    anchor = PATTERN_ANCHORS.get(pattern.anchor, 0.0)
    times = sorted({s for s, _m, _e in plan})
    out = []
    for pos, mark, entry in plan:
        if entry is None or pos < 0:
            continue
        k = times.index(pos)
        prev = times[k - 1] if k else 0
        nxt = times[k + 1] if k + 1 < len(times) else max(total_n, pos)
        zone = (pos - round(anchor * (pos - prev)), pos + round((1 - anchor) * (nxt - pos)))
        if entry.key not in renders:
            renders[entry.key] = render_clip(entry.clip, cache, bpm, domain)
        start, data = _fit(renders[entry.key], pattern.fit, pos, zone, anchor, join_n,
                           amp_cols)
        if start < 0:                                   # nothing before the block start
            data, start = data[-start:], 0
        data = data[:max(0, total_n - start)]
        if len(data) == 0:
            continue
        out.append(Clip(PulseDataSource(np.ascontiguousarray(data, dtype=np.float32)),
                        start_s=start / FS, length_s=len(data) / FS,
                        gain=velocity_gain(mark.velocity, pattern.velocity_depth),
                        name=entry.key))
    return out


def render_pattern(pattern: PatternSource, n: int, trim_n: int = 0, cache=None,
                   bpm: float = 120.0, domain: str = "pulse") -> np.ndarray:
    """*n* pulses of the pattern, skipping *trim_n* at the start."""
    from .render import mix_clips
    total = trim_n + n
    clips = placements(pattern, total, cache, bpm, domain)
    proxy = _PulseSkippingCache(cache) if cache is not None else None
    data, _active = mix_clips(clips, total, proxy, bpm, domain, pattern.overlap,
                              pattern.join_s)
    return np.ascontiguousarray(data[trim_n:], dtype=np.float32)


def pattern_key(pattern: PatternSource, domain: str, bpm: float) -> str:
    """Cache key body: everything that changes the render."""
    from .render import _generator_params, source_key
    pool = []
    for e in pattern.pool:
        c = e.clip
        src = c.source
        if isinstance(src, GeneratorSource):
            src = GeneratorSource(src.type_id, _generator_params(src, bpm), src.seed)
        sk = (source_key(src, n_pulses(c.trim_s), n_pulses(c.length_s))
              if isinstance(src, (GeneratorSource, PulseDataSource)) else src.kind)
        pool.append([e.key, e.weight, sk, c.fade_in_s, c.fade_out_s, c.gain, c.muted,
                     c.effects])
    body = json.dumps({"m": [[m.t_s, m.velocity, m.pin] for m in pattern.marks],
                       "pool": pool, "mode": pattern.mode, "seq": pattern.sequence,
                       "seed": pattern.seed, "fit": pattern.fit, "ov": pattern.overlap,
                       "j": pattern.join_s, "vd": pattern.velocity_depth, "d": domain,
                       "a": pattern.anchor,
                       "bpm": float(bpm)}, sort_keys=True, default=str)
    return "pattern:" + hashlib.blake2b(body.encode(), digest_size=16).hexdigest()

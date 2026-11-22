"""Session builder: a whole session from a few choices, deterministic per seed.

A session is laid out as phases — *Warm-up → Build 1…k → Peak → Cool-down* —
under an intensity **arc** (an amplitude envelope, ``multiply`` mode). The
arc rises in waves that grow towards the peak; the *tease* style dips just
before each crest. Generator clips (15–60 s, picked from per-style pools with
lightly varied parameters) alternate between two tracks so that their fades
overlap into crossfades. Each track gets a slew limiter as a safety stage
(no step can rise faster than ~0.17 s), phases get markers, and phase
boundaries may snap to existing markers (e.g. video scene cuts).

Qt-free: :func:`build_session` returns a :class:`SessionPlan`;
:func:`plan_tracks` turns it into model tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..generators import get_generator, list_generators
from .model import Clip, Envelope, GeneratorSource, Track

STYLES = ("mixed", "smooth", "rhythmic", "chaotic", "tease")
STYLE_LABELS = {"mixed": "Mixed (gentle → rhythmic → wild)", "smooth": "Smooth / flowing",
                "rhythmic": "Rhythmic / beats", "chaotic": "Unpredictable",
                "tease": "Tease (build-ups with drops)"}

#: generator pools per style (pulse domain)
POOLS = {
    "smooth": ("waveform", "breath", "phantom", "iso_sweep", "dense_disperse", "colored_noise",
               "noise", "sine_time"),
    "rhythmic": ("euclid", "groove", "polyrhythm", "burst", "heartbeat", "risset",
                 "beat_sweep", "penetration"),
    "chaotic": ("anti_habit", "attractor", "colored_noise", "chaos", "simplex", "noise"),
    "tease": ("touch", "rabbit", "ramp", "burst", "licks", "luxury_hj", "phantom"),
}
TRI_POOLS = {
    "smooth": ("tri_circle", "tri_spiral", "tri_lissajous", "tri_pendulum"),
    "rhythmic": ("tri_electrodes", "tri_pendulum", "tri_lissajous", "tri_rabbit"),
    "chaotic": ("tri_wander", "tri_attractor", "tri_lissajous"),
    "tease": ("tri_rabbit", "tri_electrodes", "tri_spiral", "tri_wander"),
}
#: parameters left at their defaults (levels: the arc sets intensity)
LEVEL_KEYS = ("amp", "amp_max", "amp_hit", "amp_burst", "intensity", "force", "volume",
              "peak_amp", "manual", "path", "gain")
SPEED_HINTS = ("speed", "rate", "bpm_start", "bpm_end")

LIMITER_RISE = 6.0          # /s: 0 → 1 in about 0.17 s


@dataclass
class SessionSpec:
    duration_s: float = 1200.0
    peak: float = 0.85
    style: str = "mixed"
    cycles: int = 3
    warmup_frac: float = 0.12
    cooldown_frac: float = 0.12
    triphase: bool = False
    seed: int = 0
    align_to_markers: bool = False
    bpm_sync: bool = True
    clip_min_s: float = 20.0
    clip_max_s: float = 60.0
    crossfade_s: float = 4.0


@dataclass
class Phase:
    name: str
    kind: str              # warmup | build | peak | cooldown
    t0: float
    t1: float


@dataclass
class PlannedClip:
    lane: int              # 0 / 1: which of the two tracks
    type_id: str
    params: dict
    seed: int
    start_s: float
    length_s: float
    fade_in_s: float
    fade_out_s: float
    name: str


@dataclass
class SessionPlan:
    spec: SessionSpec
    phases: list[Phase] = field(default_factory=list)
    clips: list[PlannedClip] = field(default_factory=list)
    arc: list[tuple[float, float]] = field(default_factory=list)

    @property
    def markers(self) -> list[tuple[float, str]]:
        return [(p.t0, p.name) for p in self.phases]

    def arc_at(self, times) -> np.ndarray:
        from .automation import evaluate
        return evaluate(Envelope(self.arc, "multiply", "smooth"), np.asarray(times, float))


# ── phases ──────────────────────────────────────────────────────────────────
def _phases(spec: SessionSpec, markers) -> list[Phase]:
    d = float(spec.duration_s)
    warm = d * spec.warmup_frac
    cool = d * spec.cooldown_frac
    middle = d - warm - cool
    cycles = max(1, int(spec.cycles))
    peak_len = min(middle * 0.25, 120.0)
    build_len = (middle - peak_len) / cycles
    cuts = [0.0, warm]
    for k in range(1, cycles):
        cuts.append(warm + build_len * k)
    cuts += [warm + build_len * cycles, d - cool, d]
    if spec.align_to_markers and markers:
        ms = sorted(float(m) for m in markers if 0 < m < d)
        for i in range(1, len(cuts) - 1):
            span = min(cuts[i] - cuts[i - 1], cuts[i + 1] - cuts[i])
            near = min(ms, key=lambda m: abs(m - cuts[i]), default=None)
            if near is not None and abs(near - cuts[i]) <= 0.3 * span:
                cuts[i] = near
    names = (["Warm-up"] + [f"Build {k + 1}" for k in range(cycles)] + ["Peak", "Cool-down"])
    kinds = ["warmup"] + ["build"] * cycles + ["peak", "cooldown"]
    return [Phase(n, k, cuts[i], cuts[i + 1]) for i, (n, k) in enumerate(zip(names, kinds))]


# ── the intensity arc ───────────────────────────────────────────────────────
def _arc(spec: SessionSpec, phases: list[Phase], rng) -> list[tuple[float, float]]:
    peak = float(np.clip(spec.peak, 0.05, 1.0))
    base = 0.35 * peak
    builds = [p for p in phases if p.kind == "build"]
    pts: list[tuple[float, float]] = [(0.0, 0.0)]
    warm = phases[0]
    pts.append((warm.t1, base))
    for k, ph in enumerate(builds, start=1):
        crest = peak * (0.62 + 0.3 * k / len(builds))
        trough = base + (crest - base) * 0.25
        span = ph.t1 - ph.t0
        top = ph.t0 + span * (0.7 + 0.1 * rng.random())
        pts.append((ph.t0 + span * 0.15, trough))
        if spec.style == "tease":
            pts.append((top - span * 0.08, crest * 0.95))
            pts.append((top - span * 0.04, crest * 0.35))          # the drop before the crest
            pts.append((top, crest))
        else:
            pts.append((top, crest))
        pts.append((ph.t1, trough + (crest - trough) * 0.4))
    peak_ph = next(p for p in phases if p.kind == "peak")
    span = peak_ph.t1 - peak_ph.t0
    pts.append((peak_ph.t0 + min(10.0, span * 0.2), peak))
    waves = max(1, int(span // 30))
    for w in range(waves):
        t = peak_ph.t0 + span * (w + 0.5) / waves
        pts.append((t, peak * (0.9 if w % 2 else 1.0)))
    pts.append((peak_ph.t1, peak * 0.9))
    cool = phases[-1]
    pts.append((cool.t0 + (cool.t1 - cool.t0) * 0.4, base * 0.8))
    pts.append((cool.t1, 0.0))
    pts = sorted(pts)
    out: list[tuple[float, float]] = []
    for t, v in pts:                                # strictly increasing times
        if out and t <= out[-1][0] + 1e-3:
            out[-1] = (out[-1][0], v)
        else:
            out.append((round(t, 3), round(float(np.clip(v, 0.0, 1.0)), 4)))
    return out


# ── clips ───────────────────────────────────────────────────────────────────
def _pool(spec: SessionSpec, energy: float, kind: str) -> tuple[str, ...]:
    pools = TRI_POOLS if spec.triphase else POOLS
    if spec.style == "mixed":
        if energy < 0.4:
            names = ("smooth",)
        elif energy < 0.75:
            names = ("rhythmic", "tease")
        else:
            names = ("rhythmic", "chaotic")
    else:
        names = (spec.style,)
        if kind in ("warmup", "cooldown") and spec.style != "smooth":
            names = (spec.style, "smooth")
    known = {g.type_id for g in list_generators()}
    out = []
    for n in names:
        out += [t for t in pools[n] if t in known]
    return tuple(dict.fromkeys(out))


def _params(type_id: str, energy: float, rng) -> dict:
    gen = get_generator(type_id)
    params = gen.default_params()
    for spec in gen.param_specs():
        if spec.key in LEVEL_KEYS or spec.kind not in ("float", "int"):
            continue
        if spec.min is None or spec.max is None or spec.max <= spec.min:
            continue
        lo, hi = float(spec.min), float(spec.max)
        v = float(params[spec.key])
        if any(h in spec.key for h in SPEED_HINTS):
            v *= 0.7 + 0.6 * energy                 # busier as the energy rises
        v += (rng.random() - 0.5) * 0.3 * min(hi - lo, max(abs(v), 0.1 * (hi - lo)))
        v = float(np.clip(v, lo, hi))
        params[spec.key] = int(round(v)) if spec.kind == "int" else round(v, 4)
    return params


def _clips(spec: SessionSpec, phases, plan: SessionPlan, bpm: float, rng) -> list[PlannedClip]:
    d = float(spec.duration_s)
    xf = float(spec.crossfade_s)
    bar = 4 * 60.0 / max(1e-6, bpm)
    clips: list[PlannedClip] = []
    t = 0.0
    last = None
    while t < d - 1e-6:
        length = rng.uniform(spec.clip_min_s, spec.clip_max_s)
        if spec.bpm_sync:
            length = max(bar, round(length / bar) * bar)
        end = min(d, t + length)
        if d - end < spec.clip_min_s * 0.5:          # don't leave a stub at the end
            end = d
        mid = (t + end) / 2
        energy = float(plan.arc_at([mid])[0]) / max(1e-6, spec.peak)
        kind = next((p.kind for p in phases if p.t0 <= mid < p.t1), phases[-1].kind)
        pool = [x for x in _pool(spec, energy, kind) if x != last] or list(_pool(spec, energy,
                                                                                kind))
        type_id = pool[int(rng.integers(len(pool)))]
        last = type_id
        start = max(0.0, t - (xf if clips else 0.0))
        clips.append(PlannedClip(
            lane=len(clips) % 2, type_id=type_id, params=_params(type_id, energy, rng),
            seed=int(rng.integers(0, 2**31 - 1)), start_s=round(start, 3),
            length_s=round(end - start, 3), fade_in_s=xf if clips else 0.0,
            fade_out_s=xf if end < d else 0.0, name=get_generator(type_id).label))
        t = end
    return clips


def build_session(spec: SessionSpec, bpm: float = 120.0, markers=()) -> SessionPlan:
    """Plan a session; *markers* (seconds from the session start) may attract
    phase boundaries when ``spec.align_to_markers`` is set."""
    if spec.style not in STYLES:
        raise ValueError(f"unknown style {spec.style!r}")
    if spec.duration_s < 60:
        raise ValueError("a session needs at least a minute")
    rng = np.random.default_rng(int(spec.seed))
    plan = SessionPlan(spec)
    plan.phases = _phases(spec, markers)
    plan.arc = _arc(spec, plan.phases, rng)
    plan.clips = _clips(spec, plan.phases, plan, bpm, rng)
    return plan


def plan_tracks(plan: SessionPlan, colors=("#c586c0", "#4fc1ff"),
                offset_s: float = 0.0) -> list[Track]:
    """Two model tracks holding the plan (clips, arc envelope, slew limiter),
    starting *offset_s* into the timeline."""
    from ..effects import get_effect
    tri = plan.spec.triphase
    kind = "triphase" if tri else "standard"
    tracks = []
    for lane in (0, 1):
        tr = Track(name=f"Session {'3φ ' if tri else ''}{'AB'[lane]}", type=kind,
                   color=colors[lane % len(colors)])
        keys = ("amp_l",) if tri else ("amp_l", "amp_r")
        for key in keys:
            tr.lanes[key].envelope = Envelope([(t + offset_s, v) for t, v in plan.arc],
                                              "multiply", "smooth")
        tr.effects = [get_effect("slew_limit").make(rise=LIMITER_RISE)]
        for pc in plan.clips:
            if pc.lane != lane:
                continue
            tr.clips.append(Clip(GeneratorSource(pc.type_id, dict(pc.params), pc.seed),
                                 start_s=pc.start_s + offset_s, length_s=pc.length_s,
                                 fade_in_s=pc.fade_in_s, fade_out_s=pc.fade_out_s,
                                 name=pc.name))
        tracks.append(tr)
    return tracks


def describe(plan: SessionPlan) -> str:
    from collections import Counter
    counts = Counter(c.name for c in plan.clips)
    top = ", ".join(f"{n} ×{k}" for n, k in counts.most_common(4))
    mins = plan.spec.duration_s / 60
    return (f"{len(plan.phases)} phases, {len(plan.clips)} clips over {mins:.0f} min — {top}"
            + ("…" if len(counts) > 4 else ""))

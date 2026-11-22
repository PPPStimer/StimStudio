"""Render a project to pulses (the pulse path: 40 Hz, 4 channels).

    clip source ─(cache)─► clip FX ─► clip fades/gain
        ─► track mix (Accumulator) ─► track FX ─► lane automation
        ─► master mix (blend/gain/solo) ─► master FX ─► clamp

Only the clip *source* render is cached: it is the expensive step (activities
are a per-pulse Python simulation) and it depends only on the source, the
trim and the length. Everything after it is cheap numpy and always recomputed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from ..effects import apply_effects, fade_in, fade_out
from .automation import apply_lanes
from .joins import (audible_clips, butt_joints, clip_spans, composite_gains, layout_layers,
                    smooth_joints)
from .mixer import Accumulator
from .model import (AudioFileSource, BeatMarksSource, Clip, GeneratorSource, ModRoute,
                    PatternSource, Project, PulseDataSource, Track)
from .pulse import AMP_COLS, FREQ_COLS, LANE_KEYS, n_pulses, pulse_index
from .triphase import FREQ as TRI_FREQ
from .triphase import VOL as TRI_VOL
from .triphase import Calibration, mix_tri, tri_to_pulses

log = logging.getLogger(__name__)

_AMP = list(AMP_COLS)


# ── source cache ────────────────────────────────────────────────────────────
class RenderCache:
    """Thread-safe LRU cache of rendered clip sources."""

    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            arr = self._items.get(key)
            if arr is not None:
                self._items.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return arr

    def put(self, key: str, arr: np.ndarray) -> None:
        arr.setflags(write=False)  # cached arrays are shared; never mutate them
        with self._lock:
            self._items[key] = arr
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


def source_key(source, trim_n: int, n: int, domain: str = "pulse", bpm: float = 120.0) -> str:
    if isinstance(source, PatternSource):
        from .pattern import pattern_key
        body = pattern_key(source, domain, bpm)
    elif isinstance(source, BeatMarksSource):
        body = "marks"
    elif isinstance(source, GeneratorSource):
        body = json.dumps({"t": source.type_id, "p": source.params, "s": source.seed},
                          sort_keys=True, default=str)
    elif isinstance(source, PulseDataSource):
        digest = hashlib.blake2b(np.ascontiguousarray(source.data).tobytes(),
                                 digest_size=16).hexdigest()
        body = f"pulses:{digest}:{source.loop}"
    else:
        raise TypeError(f"unknown clip source: {source!r}")
    return f"{body}|{trim_n}|{n}"


def _generator_params(source: GeneratorSource, bpm: float) -> dict:
    """The params a generator renders with; tempo-synced ones also get the BPM."""
    from ..generators import get_generator  # lazy: populates the registry
    if get_generator(source.type_id).tempo_synced:
        return {**source.params, "_bpm": float(bpm)}
    return source.params


def render_source(source, n: int, trim_n: int = 0,
                  cache: RenderCache | None = None, bpm: float = 120.0,
                  domain: str = "pulse") -> np.ndarray:
    """Render *n* pulses of a clip source, skipping *trim_n* at the start.

    Returns a read-only (n, 4) float32 array when cached; copy before editing.
    *domain* matters only to patterns (their pool renders in it); beat marks
    are silent.
    """
    if isinstance(source, GeneratorSource):
        params = _generator_params(source, bpm)
        if params is not source.params:   # tempo-synced: the BPM is part of the key
            source = GeneratorSource(source.type_id, params, source.seed)
    key = source_key(source, trim_n, n, domain, bpm) if cache is not None else None
    if key is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    if isinstance(source, PatternSource):
        from .pattern import render_pattern
        out = render_pattern(source, n, trim_n, cache, bpm, domain)
    elif isinstance(source, BeatMarksSource):
        out = np.zeros((n, 4), dtype=np.float32)
    elif isinstance(source, GeneratorSource):
        from ..generators import get_generator  # lazy: populates the registry
        gen = get_generator(source.type_id)
        out = gen.render(trim_n + n, source.params, int(source.seed)).clamp().to_array()[trim_n:]
    else:
        out = _slice_pulse_data(source.data, source.loop, trim_n, n)
    out = np.ascontiguousarray(out, dtype=np.float32)

    if key is not None:
        cache.put(key, out)
    return out


def _slice_pulse_data(data: np.ndarray, loop: bool, trim_n: int, n: int) -> np.ndarray:
    m = len(data)
    if m == 0:
        return np.zeros((n, 4), dtype=np.float32)
    if loop:
        idx = (np.arange(n) + trim_n) % m
        return data[idx]
    out = np.zeros((n, 4), dtype=np.float32)
    seg = data[trim_n:trim_n + n]
    out[:len(seg)] = seg
    if len(seg) < n:
        # past the end of the data: silence, frequency held at its last value
        last = data[min(m - 1, max(0, trim_n + len(seg) - 1))]
        out[len(seg):, list(FREQ_COLS)] = last[list(FREQ_COLS)]
    return out


# ── clip / track / project ──────────────────────────────────────────────────
def tri_effects(tri: np.ndarray, effects, t0: float, bpm: float, modulation=None) -> np.ndarray:
    """Run an effect chain on triphase data: effects see volume as amplitude and
    the pulse frequency as frequency; the position is never touched."""
    pseudo = np.stack([tri[:, TRI_VOL], tri[:, TRI_VOL], tri[:, TRI_FREQ], tri[:, TRI_FREQ]], 1)
    done = apply_effects(pseudo, effects, t0=t0, bpm=bpm, modulation=modulation)
    out = np.array(tri, dtype=np.float32)
    out[:, TRI_VOL], out[:, TRI_FREQ] = done[:, 0], done[:, 2]
    return out


def render_clip(clip: Clip, cache: RenderCache | None = None, bpm: float = 120.0,
                domain: str = "pulse", fade_in_on: bool = True,
                fade_out_on: bool = True) -> np.ndarray:
    """Render one clip to an (n, 4) float32 array (n = clip length in pulses).

    With ``domain="triphase"`` the clip holds triphase data: gain, fades and
    effects act on the volume only. *fade_in_on* / *fade_out_on* False skip a
    fade (that edge is a crossfade, see ``core.joins``).
    """
    fade_in_s = clip.fade_in_s if fade_in_on else 0.0
    fade_out_s = clip.fade_out_s if fade_out_on else 0.0
    n = n_pulses(clip.length_s)
    trim_n = n_pulses(clip.trim_s)
    out = np.array(render_source(clip.source, n, trim_n, cache, bpm, domain), dtype=np.float32)
    if domain == "triphase":
        if clip.effects:
            out = tri_effects(out, clip.effects, clip.start_s, bpm)
        out[:, TRI_VOL] *= float(clip.gain)
        fade_in(out, fade_in_s, channels=("A",))
        fade_out(out, fade_out_s, channels=("A",))
        np.clip(out, 0.0, 1.0, out=out)
        return out
    if clip.effects:
        out = apply_effects(out, clip.effects, t0=clip.start_s, bpm=bpm)
        out = np.array(out, dtype=np.float32)
    if clip.gain != 1.0:
        out[:, _AMP] *= float(clip.gain)
    fade_in(out, fade_in_s)
    fade_out(out, fade_out_s)
    np.clip(out, 0.0, 1.0, out=out)
    return out


@dataclass
class TrackRender:
    pulses: np.ndarray   # (N, 4) float32, before track gain; muted amplitudes are 0
    active: np.ndarray   # (N,) bool — where the track has clip content
    #: audio_mod tracks: modulation signals over the timeline (name → (N,) 0..1)
    signals: dict | None = None
    #: triphase tracks: the triphase data (volume, x, y, freq) before conversion
    triphase: np.ndarray | None = None
    #: (N, 2) amplitude before the lane mutes: weights the frequency in the mix
    freq_weight: np.ndarray | None = None
    #: triphase tracks: (N,) volume before the lane mute, as *freq_weight*
    tri_weight: np.ndarray | None = None
    #: per side (A, B): does the track's amplitude / frequency reach the master
    amp_on: tuple[bool, bool] = (True, True)
    freq_on: tuple[bool, bool] = (True, True)


# ── audio modulation signals ────────────────────────────────────────────────
def clip_signals(clip: Clip) -> dict[str, np.ndarray]:
    """The analysis signals under one audio clip (trim, gain and fades applied).

    Analyses the file on first use (cached per file). A missing or unreadable
    file renders as silence.
    """
    from ..audio.analysis import SIGNALS, analyze_file, silent_signals
    n = n_pulses(clip.length_s)
    trim = n_pulses(clip.trim_s)
    try:
        full = analyze_file(clip.source.path)
    except Exception as exc:  # noqa: BLE001
        log.warning("audio clip %r: %s", clip.source.path, exc)
        return silent_signals(n)
    ramp = np.ones(n, dtype=np.float32) * float(clip.gain)
    k_in, k_out = min(n, n_pulses(clip.fade_in_s)), min(n, n_pulses(clip.fade_out_s))
    if k_in:
        ramp[:k_in] *= np.linspace(0.0, 1.0, k_in)
    if k_out:
        ramp[n - k_out:] *= np.linspace(1.0, 0.0, k_out)
    out = {}
    for name in SIGNALS:
        seg = np.zeros(n, dtype=np.float32)
        src = full[name][trim:trim + n]
        seg[:len(src)] = src
        out[name] = np.clip(seg * ramp, 0.0, 1.0)
    return out


def render_audio_track(track: Track, n_total: int) -> TrackRender:
    from ..audio.analysis import silent_signals
    signals = silent_signals(n_total)
    active = np.zeros(n_total, dtype=bool)
    for clip in track.clips:
        if clip.muted or not isinstance(clip.source, AudioFileSource) or clip.length_s <= 0:
            continue
        s0 = pulse_index(clip.start_s)
        lo, hi = max(0, s0), min(n_total, s0 + n_pulses(clip.length_s))
        if hi <= lo:
            continue
        for name, seg in clip_signals(clip).items():
            np.maximum(signals[name][lo:hi], seg[lo - s0:hi - s0], out=signals[name][lo:hi])
        active[lo:hi] = True
    if track.gain != 1.0:
        signals = {k: np.clip(v * float(track.gain), 0, 1) for k, v in signals.items()}
    return TrackRender(np.zeros((n_total, 4), np.float32), active, signals)


def smooth_signal(s: np.ndarray, seconds: float) -> np.ndarray:
    """Causal exponential smoothing with time constant *seconds*."""
    if seconds <= 0 or len(s) == 0:
        return s
    tau = seconds * 40.0
    k = np.exp(-np.arange(int(tau * 5) + 1) / tau)
    k /= k.sum()
    return np.convolve(s, k)[:len(s)]


def route_signal(route: ModRoute, signals: dict) -> np.ndarray | None:
    """The route's source signal (smoothed / inverted), or None if unavailable."""
    src = signals.get(route.source_track)
    if not src or route.signal not in src:
        return None
    s = smooth_signal(np.asarray(src[route.signal], dtype=np.float64), route.smooth_s)
    return 1.0 - s if route.invert else s


def apply_mod(x, s, mode: str, depth: float):
    """Combine value(s) *x* (0..1) with signal *s* (see :class:`ModRoute`)."""
    if mode == "add":
        return x + depth * s
    if mode == "replace":
        return x * (1.0 - depth) + depth * s
    return x * (1.0 - depth + depth * s)    # multiply


def _track_modulation(track: Track, routes, signals):
    """Split the routes aimed at *track* into lane and effect-parameter parts."""
    lanes: list[tuple[int, np.ndarray, ModRoute]] = []
    fx: dict[int, dict[str, list]] = {}
    for route in routes:
        if not route.enabled or route.target_track != track.id:
            continue
        s = route_signal(route, signals)
        if s is None:
            continue
        kind, _, rest = route.target.partition(":")
        if kind == "lane" and rest in LANE_KEYS:
            lanes.append((LANE_KEYS.index(rest), s, route))
        elif kind == "fx":
            index, _, key = rest.partition(":")
            if index.isdigit() and key:
                fn = (lambda u, s=s, r=route: apply_mod(u, s, r.mode, r.depth))
                fx.setdefault(int(index), {}).setdefault(key, []).append(fn)
    return lanes, fx

def mix_clips(clips, n_total: int, cache: RenderCache | None = None, bpm: float = 120.0,
              domain: str = "pulse", overlap: str = "mix", join_s: float = 0.25,
              smooth: bool = False, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Combine the clips of one track into ``(data, active)`` over *n_total*
    pulses starting at timeline pulse *offset*: (n, 4) pulses, or triphase data
    with ``domain="triphase"`` (volumes add, position and frequency are
    volume-weighted). *overlap*, *join_s* and *smooth* are the track's settings
    (see ``core.joins``). Muted clips take no part."""
    clips = audible_clips(clips)
    spans = clip_spans(clips, offset)
    join_n = n_pulses(join_s)
    layers = layout_layers(spans, join_n) if overlap == "crossfade" else None
    gains = composite_gains(layers) if layers else None
    blend = "max" if overlap == "max" else "add"
    tri = domain == "triphase"
    acc = Accumulator(n_total)      # pulse: the mix · triphase: fed (vol, vol, x, y)
    frq = Accumulator(n_total) if tri else None     # triphase: fed (vol, vol, f, f)
    for k, clip in enumerate(clips):
        cov_in = layers[k].covered_in if layers else False
        cov_out = layers[k].covered_out if layers else False
        data = render_clip(clip, cache, bpm, domain, not cov_in, not cov_out)
        start = spans[k][0]
        if gains is not None:
            g = gains[k][:len(data), None].astype(np.float32)
            cols = [TRI_VOL] if tri else _AMP
            data[:, cols] *= g
        if tri:
            v = data[:, TRI_VOL]
            acc.add(np.stack([v, v, data[:, 1], data[:, 2]], 1), start, blend)
            frq.add(np.stack([v, v, data[:, TRI_FREQ], data[:, TRI_FREQ]], 1), start, blend)
        else:
            acc.add(data, start, blend)
    out, active = acc.result()
    if tri:
        f_out, _ = frq.result()
        out = np.stack([out[:, 0], out[:, 2], out[:, 3], f_out[:, 2]], 1).astype(np.float32)
    if smooth and join_n > 1:
        smooth_joints(out, butt_joints(spans), join_n // 2, range(4), spans)
    return out, active


def render_triphase_track(track: Track, n_total: int, cache=None, bpm: float = 120.0,
                          routes=(), signals=None, cal: Calibration | None = None) -> TrackRender:
    """Triphase track: clips mix (see :func:`mix_clips`), then track FX (volume and frequency), lanes, modulation, and the
    conversion to A/B pulses for the pulse outputs."""
    cal = cal or Calibration()
    tri, active = mix_clips(track.clips, n_total, cache, bpm, "triphase", track.overlap,
                            track.join_s, track.smooth_joins)

    lane_mods, fx_mods = _track_modulation(track, routes, signals or {})
    if track.effects:
        tri = tri_effects(tri, track.effects, 0.0, bpm, fx_mods)
    apply_lanes(tri, track.lanes, mute=False)
    for col, s, route in lane_mods:
        tri[:, col] = np.clip(apply_mod(tri[:, col], s, route.mode, route.depth), 0.0, 1.0)
    tri[~active, TRI_VOL] = 0.0
    pulses = tri_to_pulses(tri, cal)
    weight, tri_weight = pulses[:, _AMP].copy(), tri[:, TRI_VOL].copy()
    # only Volume and Pulse freq can be muted; position always follows
    vol_on = not track.lanes["amp_l"].muted
    freq_on = not track.lanes["freq_r"].muted
    if not vol_on:
        tri[:, TRI_VOL] = 0.0
        pulses[:, _AMP] = 0.0
    return TrackRender(pulses, active, triphase=tri, freq_weight=weight, tri_weight=tri_weight,
                       amp_on=(vol_on, vol_on), freq_on=(freq_on, freq_on))


def render_track(track: Track, n_total: int, cache: RenderCache | None = None,
                 bpm: float = 120.0, routes=(), signals=None,
                 cal: Calibration | None = None) -> TrackRender:
    """Render a track over the whole timeline (``n_total`` pulses).

    *routes* / *signals*: modulation routes of the project and the signals of
    its audio_mod tracks (``track id → name → array``).
    """
    if track.type == "audio_mod":
        return render_audio_track(track, n_total)
    if track.type == "triphase":
        return render_triphase_track(track, n_total, cache, bpm, routes, signals, cal)
    if track.type != "standard":
        log.debug("track %r: type %r is not rendered yet", track.name, track.type)
        return TrackRender(np.zeros((n_total, 4), np.float32), np.zeros(n_total, bool))

    pulses, active = mix_clips(track.clips, n_total, cache, bpm, "pulse", track.overlap,
                               track.join_s, track.smooth_joins)

    lane_mods, fx_mods = _track_modulation(track, routes, signals or {})
    if track.effects:
        pulses = np.array(apply_effects(pulses, track.effects, t0=0.0, bpm=bpm,
                                        modulation=fx_mods), dtype=np.float32)
    apply_lanes(pulses, track.lanes, mute=False)
    for col, s, route in lane_mods:
        pulses[:, col] = np.clip(apply_mod(pulses[:, col], s, route.mode, route.depth), 0.0, 1.0)
    # a track is silent where it has no clips, whatever its effects produced
    pulses[np.ix_(~active, _AMP)] = 0.0
    # lane mutes come last so modulation can't bring a muted side back
    weight = pulses[:, _AMP].copy()
    amp_on = tuple(not track.lanes[k].muted for k in ("amp_l", "amp_r"))
    freq_on = tuple(not track.lanes[k].muted for k in ("freq_l", "freq_r"))
    for col, on in zip(_AMP, amp_on):
        if not on:
            pulses[:, col] = 0.0
    return TrackRender(pulses, active, freq_weight=weight, amp_on=amp_on, freq_on=freq_on)


def audible_tracks(project: Project) -> list[Track]:
    """Tracks that reach the master, honouring mute and solo."""
    tracks = [t for t in project.tracks if t.type != "video"]    # video is visual only
    soloed = [t for t in tracks if t.solo and not t.muted]
    if soloed:
        return soloed
    return [t for t in tracks if not t.muted]


@dataclass
class ProjectRender:
    #: (N, 4) — every track, triphase tracks converted to A/B: .hwl, Howl, wavelet
    master: np.ndarray
    tracks: dict[str, TrackRender] = field(default_factory=dict)
    #: audio_mod track id → signal name → (N,) array
    signals: dict[str, dict] = field(default_factory=dict)
    #: (N, 4) — standard tracks only; continuous audio adds ``triphase`` with phase
    master_audio: np.ndarray | None = None
    #: (N, 4) combined triphase data of the audible triphase tracks, or None
    triphase: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.master)


def render_project(project: Project, cache: RenderCache | None = None) -> ProjectRender:
    """Render every track and the master mix."""
    n_total = project.n_pulses()
    renders: dict[str, TrackRender] = {}
    signals: dict[str, dict] = {}
    # modulation sources first: audio_mod tracks only produce signals
    for track in project.tracks:
        if track.type == "audio_mod":
            renders[track.id] = render_audio_track(track, n_total)
            if not track.muted:
                signals[track.id] = renders[track.id].signals
    cal = Calibration.from_dict(project.triphase_cal)
    acc = Accumulator(n_total)          # everything (pulse outputs)
    acc_audio = Accumulator(n_total)    # standard tracks only (continuous audio)
    tri_layers = []
    audible = {t.id for t in audible_tracks(project) if t.type != "audio_mod"}
    for track in project.tracks:
        if track.type in ("audio_mod", "video"):
            continue
        tr = render_track(track, n_total, cache, project.bpm, project.mod_routes, signals, cal)
        renders[track.id] = tr
        if track.id not in audible:
            continue
        mix = dict(freq_weight=tr.freq_weight, amp_on=tr.amp_on, freq_on=tr.freq_on)
        acc.add(tr.pulses, 0, track.blend, track.gain, tr.active, **mix)
        if tr.triphase is not None:
            tri_layers.append((tr.triphase, track.gain, tr.tri_weight, tr.freq_on[0]))
        else:
            acc_audio.add(tr.pulses, 0, track.blend, track.gain, tr.active, **mix)
    master, _active = acc.result()
    master_audio, _ = acc_audio.result()
    triphase = mix_tri(tri_layers)
    fx = project.master_effects
    if fx:
        master = np.array(apply_effects(master, fx, t0=0.0, bpm=project.bpm), dtype=np.float32)
        master_audio = np.array(apply_effects(master_audio, fx, t0=0.0, bpm=project.bpm),
                                dtype=np.float32)
        if triphase is not None:   # master safety (slew limit, scale) reaches triphase too
            triphase = tri_effects(triphase, fx, 0.0, project.bpm)
    np.clip(master, 0.0, 1.0, out=master)
    np.clip(master_audio, 0.0, 1.0, out=master_audio)
    return ProjectRender(master, renders, signals, master_audio, triphase)


def render_master(project: Project, cache: RenderCache | None = None) -> np.ndarray:
    """The final (N, 4) pulse array — what goes to .hwl / Howl."""
    return render_project(project, cache).master


# ── activity stages (Howl activities run through internal stages) ───────────
def segment_boundaries(type_id: str, params: dict, seed: int, horizon_s: float) -> list[float]:
    """Times (s) where the generator's internal stage changes, up to *horizon_s*."""
    from ..generators import get_generator
    return get_generator(type_id).segment_boundaries(n_pulses(horizon_s), params, int(seed))


def segment_stages(type_id: str, params: dict, seed: int,
                   horizon_s: float) -> list[tuple[float, float]]:
    """Stage spans ``(start, end)``; empty if the generator has no stages. The
    last span is cut at *horizon_s*."""
    bounds = segment_boundaries(type_id, params, seed, horizon_s)
    if not bounds:
        return []
    edges = [0.0, *bounds, float(horizon_s)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def manual_params_at(type_id: str, params: dict, seed: int, at_s: float) -> dict:
    """The activity's manual-mode parameters at *at_s* (empty if not supported)."""
    from ..generators import get_generator
    return get_generator(type_id).manual_params_at(n_pulses(at_s), params, int(seed))

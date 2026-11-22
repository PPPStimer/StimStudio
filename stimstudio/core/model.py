"""Project data model: Project → Track → Clip, plus lane automation.

The model is plain data. Rendering lives in ``core.render``, persistence in
``io.project_file``. Times are seconds on the project timeline; a pulse index
is ``round(t * FS)``.

Track types:

* ``standard`` — four lanes (``amp_l, amp_r, freq_l, freq_r``) filled by clips.
* ``audio_mod`` — audio clips analysed into modulation signals (envelope,
  onsets, bands). Silent on its own; ``ModRoute`` entries route its signals
  to lanes or effect parameters of other tracks.
* ``triphase`` — clips produce triphase data (volume, position x/y, freq; see
  ``core.triphase``); lanes are Volume / Position X / Position Y / Pulse freq.
* ``video`` — one video clip shown in sync with the timeline. Purely visual:
  never rendered, the clip can only be moved.

Standard and triphase tracks also take two rhythm sources: a silent
``BeatMarksSource`` (just beat times) and a ``PatternSource`` that plays a
clip of its pool at every mark (``core.pattern``).
"""

from __future__ import annotations

import base64
import copy
import uuid
from dataclasses import dataclass, field
from typing import Union

import numpy as np

from .pulse import FS, LANE_KEYS, array_from_bytes, array_to_bytes

#: 2: marks / pattern clip sources · 3: frequency lane mutes
SCHEMA_VERSION = 3

TRACK_TYPES = ("standard", "triphase", "audio_mod", "video")
BLEND_MODES = ("add", "max", "multiply", "replace")
ENVELOPE_MODES = ("multiply", "offset", "replace")
ENVELOPE_INTERPS = ("linear", "hold", "smooth")
MOD_MODES = ("multiply", "add", "replace")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def entry_address(clip_id: str, key: str) -> str:
    """The ``Project.find_clip`` address of a pattern pool entry's clip."""
    return f"{clip_id}/{key}"


# ── automation ──────────────────────────────────────────────────────────────
@dataclass
class Envelope:
    """Automation curve on one lane. Points are ``(time_s, value 0..1)``.

    *mode* says how the curve combines with the lane signal:
    ``multiply`` (lane × v), ``offset`` (lane + (2v − 1)), ``replace`` (v).
    """

    points: list[tuple[float, float]] = field(default_factory=list)
    mode: str = "multiply"
    interp: str = "linear"
    enabled: bool = True

    def to_dict(self) -> dict:
        return {"points": [[float(t), float(v)] for t, v in self.points],
                "mode": self.mode, "interp": self.interp, "enabled": self.enabled}

    def clone(self) -> "Envelope":
        return Envelope(list(self.points), self.mode, self.interp, self.enabled)

    @classmethod
    def from_dict(cls, d: dict) -> "Envelope":
        return cls(points=[(float(t), float(v)) for t, v in d.get("points", [])],
                   mode=d.get("mode", "multiply"), interp=d.get("interp", "linear"),
                   enabled=d.get("enabled", True))


@dataclass
class Lane:
    """One of a track's sub-lanes. Muting an amplitude lane silences that side;
    muting a frequency lane stops the track steering that side's frequency
    (see ``core.mixer``). Triphase tracks mute only Volume and Pulse freq."""

    muted: bool = False
    envelope: Envelope | None = None

    def to_dict(self) -> dict:
        return {"muted": self.muted,
                "envelope": self.envelope.to_dict() if self.envelope else None}

    def clone(self) -> "Lane":
        return Lane(self.muted, self.envelope.clone() if self.envelope else None)

    @classmethod
    def from_dict(cls, d: dict) -> "Lane":
        env = d.get("envelope")
        return cls(muted=d.get("muted", False),
                   envelope=Envelope.from_dict(env) if env else None)


# ── clip sources ────────────────────────────────────────────────────────────
@dataclass
class GeneratorSource:
    """Clip content rendered by a registered generator (same as a .parthwl part)."""

    type_id: str
    params: dict = field(default_factory=dict)
    seed: int = 0

    kind = "generator"

    def clone(self) -> "GeneratorSource":
        return GeneratorSource(self.type_id, copy.deepcopy(self.params), self.seed)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "type": self.type_id,
                "params": dict(self.params), "seed": int(self.seed)}


@dataclass
class PulseDataSource:
    """Clip content stored as raw pulses (imported .hwl, converted audio, …)."""

    data: np.ndarray  # (N, 4) float32
    loop: bool = False
    origin: str = ""  # free text: where the data came from

    kind = "pulses"

    def clone(self) -> "PulseDataSource":
        # the data array is shared: pulse data is never edited in place
        return PulseDataSource(self.data, self.loop, self.origin)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "loop": self.loop, "origin": self.origin,
                "encoding": "base64-f32le", "n": int(len(self.data)),
                "data": base64.b64encode(array_to_bytes(self.data)).decode("ascii")}


@dataclass
class AudioFileSource:
    """An audio file on an ``audio_mod`` track (analysed, never played)."""

    path: str

    kind = "audio"

    def clone(self) -> "AudioFileSource":
        return AudioFileSource(self.path)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "path": self.path}


@dataclass
class VideoFileSource:
    """A video file on a ``video`` track (shown, never rendered)."""

    path: str
    duration_s: float = 0.0

    kind = "video"

    def clone(self) -> "VideoFileSource":
        return VideoFileSource(self.path, self.duration_s)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "path": self.path, "duration_s": self.duration_s}


@dataclass
class Mark:
    """A beat mark inside a marks / pattern clip. *t_s* is source time (like
    ``Clip.trim_s``: 0 is the start of the source); *pin* is the key of the
    pool entry that always plays here (pattern clips only)."""

    t_s: float
    velocity: float = 1.0
    pin: str | None = None
    id: str = field(default_factory=new_id)

    def clone(self) -> "Mark":
        return Mark(self.t_s, self.velocity, self.pin, self.id)

    def to_dict(self) -> dict:
        return {"id": self.id, "t_s": float(self.t_s), "velocity": float(self.velocity),
                "pin": self.pin}

    @classmethod
    def from_dict(cls, d: dict) -> "Mark":
        pin = d.get("pin")
        return cls(t_s=float(d.get("t_s", 0.0)), velocity=float(d.get("velocity", 1.0)),
                   pin=str(pin) if pin else None, id=d.get("id") or new_id())


def _marks_from(items) -> list[Mark]:
    return sorted((Mark.from_dict(m) for m in items), key=lambda m: m.t_s)


@dataclass
class BeatMarksSource:
    """A blank of beat marks: silent, it only holds times (imported beats or
    peaks found in a clip) until it is turned into a :class:`PatternSource`."""

    marks: list[Mark] = field(default_factory=list)

    kind = "marks"

    def clone(self) -> "BeatMarksSource":
        return BeatMarksSource([m.clone() for m in self.marks])

    def to_dict(self) -> dict:
        return {"kind": self.kind, "marks": [m.to_dict() for m in self.marks]}


@dataclass
class PoolEntry:
    """One clip a pattern can play, stored as an embedded copy (its
    ``start_s`` is ignored; length, trim, fades, gain and effects count)."""

    key: str
    clip: "Clip"
    weight: float = 1.0

    def clone(self) -> "PoolEntry":
        return PoolEntry(self.key, self.clip.clone(), self.weight)

    def to_dict(self) -> dict:
        return {"key": self.key, "weight": float(self.weight), "clip": self.clip.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "PoolEntry":
        return cls(key=str(d.get("key", "A")), clip=Clip.from_dict(d["clip"]),
                   weight=float(d.get("weight", 1.0)))


PATTERN_MODES = ("random", "sequence")
PATTERN_FITS = ("cut", "own", "loop", "stretch")
#: which point of a clip lands on its mark, as a fraction of the clip
PATTERN_ANCHORS = {"start": 0.0, "center": 0.5, "end": 1.0}


@dataclass
class PatternSource:
    """A container: at every mark one clip of the pool plays (see
    ``core.pattern``). *mode* ``random`` picks by weight (seeded),
    ``sequence`` walks *sequence* (pool keys separated by spaces). *fit* says
    how a clip fills the time around its mark; *anchor* which point of the
    clip (start / center / end) lands on the mark; *overlap* / *join_s* work
    like the track settings of the same name, between the clips inside."""

    marks: list[Mark] = field(default_factory=list)
    pool: list[PoolEntry] = field(default_factory=list)
    mode: str = "random"
    sequence: str = ""
    seed: int = 0
    fit: str = "cut"
    overlap: str = "max"
    join_s: float = 0.05
    velocity_depth: float = 1.0
    anchor: str = "start"

    kind = "pattern"

    def clone(self) -> "PatternSource":
        return PatternSource([m.clone() for m in self.marks], [e.clone() for e in self.pool],
                             self.mode, self.sequence, self.seed, self.fit, self.overlap,
                             self.join_s, self.velocity_depth, self.anchor)

    def entry(self, key: str) -> PoolEntry | None:
        return next((e for e in self.pool if e.key == key), None)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "marks": [m.to_dict() for m in self.marks],
                "pool": [e.to_dict() for e in self.pool], "mode": self.mode,
                "sequence": self.sequence, "seed": int(self.seed), "fit": self.fit,
                "overlap": self.overlap, "join_s": float(self.join_s),
                "velocity_depth": float(self.velocity_depth), "anchor": self.anchor}


ClipSource = Union[GeneratorSource, PulseDataSource, AudioFileSource, VideoFileSource,
                   BeatMarksSource, PatternSource]


def source_from_dict(d: dict) -> ClipSource:
    kind = d.get("kind")
    if kind == "marks":
        return BeatMarksSource(_marks_from(d.get("marks", [])))
    if kind == "pattern":
        return PatternSource(
            marks=_marks_from(d.get("marks", [])),
            pool=[PoolEntry.from_dict(e) for e in d.get("pool", [])],
            mode=str(d.get("mode", "random")), sequence=str(d.get("sequence", "")),
            seed=int(d.get("seed", 0)), fit=str(d.get("fit", "cut")),
            overlap=str(d.get("overlap", "max")), join_s=float(d.get("join_s", 0.05)),
            velocity_depth=float(d.get("velocity_depth", 1.0)),
            anchor=str(d.get("anchor", "start")))
    if kind == "audio":
        return AudioFileSource(str(d.get("path", "")))
    if kind == "video":
        return VideoFileSource(str(d.get("path", "")), float(d.get("duration_s", 0.0)))
    if kind == "generator":
        return GeneratorSource(d["type"], dict(d.get("params", {})), int(d.get("seed", 0)))
    if kind == "pulses":
        data = array_from_bytes(base64.b64decode(d["data"]))
        return PulseDataSource(data, bool(d.get("loop", False)), d.get("origin", ""))
    raise ValueError(f"unknown clip source kind: {kind!r}")


# ── clip / track / project ──────────────────────────────────────────────────
@dataclass
class Clip:
    source: ClipSource
    start_s: float = 0.0
    length_s: float = 10.0
    #: Seconds skipped at the start of the source (generators still simulate
    #: from 0, so stateful activities stay deterministic).
    trim_s: float = 0.0
    fade_in_s: float = 0.0
    fade_out_s: float = 0.0
    gain: float = 1.0
    muted: bool = False
    effects: list[dict] = field(default_factory=list)
    name: str = ""
    id: str = field(default_factory=new_id)

    @property
    def end_s(self) -> float:
        return self.start_s + self.length_s

    def clone(self, fresh_id: bool = False) -> "Clip":
        """Independent copy (pulse data shared). *fresh_id* gives it a new id."""
        return Clip(self.source.clone(), self.start_s, self.length_s, self.trim_s,
                    self.fade_in_s, self.fade_out_s, self.gain, self.muted,
                    copy.deepcopy(self.effects), self.name,
                    new_id() if fresh_id else self.id)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "start_s": self.start_s,
                "length_s": self.length_s, "trim_s": self.trim_s,
                "fade_in_s": self.fade_in_s, "fade_out_s": self.fade_out_s,
                "gain": self.gain, "muted": self.muted,
                "effects": [dict(e) for e in self.effects],
                "source": self.source.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        return cls(source=source_from_dict(d["source"]),
                   start_s=float(d.get("start_s", 0.0)), length_s=float(d.get("length_s", 10.0)),
                   trim_s=float(d.get("trim_s", 0.0)),
                   fade_in_s=float(d.get("fade_in_s", 0.0)),
                   fade_out_s=float(d.get("fade_out_s", 0.0)),
                   gain=float(d.get("gain", 1.0)), muted=bool(d.get("muted", False)),
                   effects=[dict(e) for e in d.get("effects", [])],
                   name=d.get("name", ""), id=d.get("id") or new_id())


def _default_lanes() -> dict[str, Lane]:
    return {key: Lane() for key in LANE_KEYS}


def _mute_freq_with_amp(track: "Track") -> None:
    """Projects before version 3: a muted amplitude lane also took the side's
    frequency out of the mix (its weight was ~0), so mute that frequency too."""
    pairs = ((("amp_l", "freq_r"),) if track.type == "triphase"
             else (("amp_l", "freq_l"), ("amp_r", "freq_r")))
    for amp, freq in pairs:
        if track.lanes[amp].muted:
            track.lanes[freq].muted = True


@dataclass
class Track:
    name: str = "Track"
    type: str = "standard"
    clips: list[Clip] = field(default_factory=list)
    effects: list[dict] = field(default_factory=list)
    lanes: dict[str, Lane] = field(default_factory=_default_lanes)
    gain: float = 1.0
    muted: bool = False
    solo: bool = False
    blend: str = "add"
    color: str = "#4ec9b0"
    id: str = field(default_factory=new_id)
    #: how overlapping clips combine (``core.joins.OVERLAP_MODES``); projects
    #: saved before the field existed load as "mix" and sound unchanged
    overlap: str = "crossfade"
    #: ramp length for nested clips and for smoothing butt joints (s)
    join_s: float = 0.25
    #: round off butt joints (one clip ending where the next begins)
    smooth_joins: bool = False

    def end_s(self) -> float:
        return max((c.end_s for c in self.clips), default=0.0)

    def clone(self, new_ids: bool = False) -> "Track":
        return Track(self.name, self.type, [c.clone(new_ids) for c in self.clips],
                     copy.deepcopy(self.effects),
                     {k: lane.clone() for k, lane in self.lanes.items()},
                     self.gain, self.muted, self.solo, self.blend, self.color,
                     new_id() if new_ids else self.id, self.overlap, self.join_s,
                     self.smooth_joins)

    def find_clip(self, clip_id: str) -> Clip | None:
        return next((c for c in self.clips if c.id == clip_id), None)

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "type": self.type,
                "gain": self.gain, "muted": self.muted, "solo": self.solo,
                "blend": self.blend, "color": self.color,
                "overlap": self.overlap, "join_s": self.join_s,
                "smooth_joins": self.smooth_joins,
                "effects": [dict(e) for e in self.effects],
                "lanes": {k: lane.to_dict() for k, lane in self.lanes.items()},
                "clips": [c.to_dict() for c in self.clips]}

    @classmethod
    def from_dict(cls, d: dict) -> "Track":
        lanes = _default_lanes()
        for key, ld in d.get("lanes", {}).items():
            lanes[key] = Lane.from_dict(ld)
        return cls(name=d.get("name", "Track"), type=d.get("type", "standard"),
                   clips=[Clip.from_dict(c) for c in d.get("clips", [])],
                   effects=[dict(e) for e in d.get("effects", [])], lanes=lanes,
                   gain=float(d.get("gain", 1.0)), muted=bool(d.get("muted", False)),
                   solo=bool(d.get("solo", False)), blend=d.get("blend", "add"),
                   color=d.get("color", "#4ec9b0"), id=d.get("id") or new_id(),
                   overlap=str(d.get("overlap", "mix")),
                   join_s=float(d.get("join_s", 0.25)),
                   smooth_joins=bool(d.get("smooth_joins", False)))


@dataclass
class ModRoute:
    """Route one signal of an ``audio_mod`` track to a target on another track.

    *target* is ``"lane:<lane key>"`` or ``"fx:<effect index>:<param>"``. With
    the signal ``s`` (0..1, optionally smoothed / inverted) and depth ``d`` a
    value ``x`` (lane value, or the parameter's 0..1 position) becomes:
    ``multiply`` x·(1−d+d·s) · ``add`` x+d·s (d may be negative) · ``replace``
    x·(1−d)+d·s.
    """

    source_track: str
    signal: str = "envelope"
    target_track: str = ""
    target: str = "lane:amp_l"
    mode: str = "multiply"
    depth: float = 1.0
    smooth_s: float = 0.0
    invert: bool = False
    enabled: bool = True
    id: str = field(default_factory=new_id)

    def clone(self) -> "ModRoute":
        return ModRoute(**{**self.__dict__})

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "ModRoute":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


@dataclass
class Marker:
    """A named point on the timeline (cue, scene cut, session phase)."""

    t_s: float
    name: str = ""
    color: str = "#dcdcaa"
    id: str = field(default_factory=new_id)

    def clone(self) -> "Marker":
        return Marker(self.t_s, self.name, self.color, self.id)

    def to_dict(self) -> dict:
        return {"id": self.id, "t_s": self.t_s, "name": self.name, "color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "Marker":
        return cls(t_s=float(d.get("t_s", 0.0)), name=str(d.get("name", "")),
                   color=str(d.get("color", "#dcdcaa")), id=d.get("id") or new_id())


@dataclass
class Project:
    name: str = "Untitled"
    tracks: list[Track] = field(default_factory=list)
    master_effects: list[dict] = field(default_factory=list)
    bpm: float = 120.0
    #: Fixed length in seconds; ``None`` = end of the last clip.
    length_s: float | None = None
    #: Default output frequency range (Hz) used to show 0..1 lanes in Hz.
    freq_range_hz: tuple[float, float] = (10.0, 100.0)
    meta: dict = field(default_factory=dict)
    mod_routes: list[ModRoute] = field(default_factory=list)
    #: triphase calibration (``core.triphase.Calibration`` fields)
    triphase_cal: dict = field(default_factory=dict)
    #: timeline markers, kept sorted by time
    markers: list[Marker] = field(default_factory=list)

    def duration_s(self) -> float:
        if self.length_s is not None:
            return float(self.length_s)
        return max((t.end_s() for t in self.tracks), default=0.0)

    def n_pulses(self) -> int:
        return max(0, round(self.duration_s() * FS))

    def clone(self) -> "Project":
        """Snapshot for background rendering (pulse data arrays are shared)."""
        return Project(self.name, [t.clone() for t in self.tracks],
                       copy.deepcopy(self.master_effects), self.bpm, self.length_s,
                       tuple(self.freq_range_hz), copy.deepcopy(self.meta),
                       [r.clone() for r in self.mod_routes], dict(self.triphase_cal),
                       [m.clone() for m in self.markers])

    def find_clip(self, clip_id: str) -> tuple[Track, Clip] | None:
        """``(track, clip)`` by id. ``"<pattern clip id>/<pool key>"`` addresses
        the clip of a pattern's pool entry (the track is the pattern's)."""
        if "/" in clip_id:
            parent_id, key = clip_id.split("/", 1)
            found = self.find_clip(parent_id)
            if found is None or not isinstance(found[1].source, PatternSource):
                return None
            entry = found[1].source.entry(key)
            return (found[0], entry.clip) if entry is not None else None
        for t in self.tracks:
            c = t.find_clip(clip_id)
            if c is not None:
                return t, c
        return None

    def video_track(self) -> Track | None:
        return next((t for t in self.tracks if t.type == "video"), None)

    def track_by_id(self, track_id: str) -> Track:
        for t in self.tracks:
            if t.id == track_id:
                return t
        raise KeyError(track_id)

    def to_dict(self) -> dict:
        return {"format": "stimproj", "version": SCHEMA_VERSION, "fs": FS,
                "name": self.name, "bpm": self.bpm, "length_s": self.length_s,
                "freq_range_hz": list(self.freq_range_hz), "meta": dict(self.meta),
                "master_effects": [dict(e) for e in self.master_effects],
                "tracks": [t.to_dict() for t in self.tracks],
                "mod_routes": [r.to_dict() for r in self.mod_routes],
                "triphase_cal": dict(self.triphase_cal),
                "markers": [m.to_dict() for m in self.markers]}

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        if d.get("format") != "stimproj":
            raise ValueError("not a StimStudio project (wrong or missing 'format')")
        if int(d.get("version", 1)) > SCHEMA_VERSION:
            raise ValueError(f"project version {d['version']} is newer than this app supports")
        lo, hi = d.get("freq_range_hz", (10.0, 100.0))
        tracks = [Track.from_dict(t) for t in d.get("tracks", [])]
        if int(d.get("version", 1)) < 3:
            for t in tracks:
                _mute_freq_with_amp(t)
        return cls(name=d.get("name", "Untitled"), tracks=tracks,
                   master_effects=[dict(e) for e in d.get("master_effects", [])],
                   bpm=float(d.get("bpm", 120.0)), length_s=d.get("length_s"),
                   freq_range_hz=(float(lo), float(hi)), meta=dict(d.get("meta", {})),
                   mod_routes=[ModRoute.from_dict(r) for r in d.get("mod_routes", [])],
                   triphase_cal=dict(d.get("triphase_cal", {})),
                   markers=sorted((Marker.from_dict(m) for m in d.get("markers", [])),
                                  key=lambda m: m.t_s))

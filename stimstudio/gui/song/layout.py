"""Vertical layout of the song editor and the snap grid."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...core.model import Project

CLIP_ROW_H = 58      # main row (clips) of a track
LANE_H = 40          # each expanded sub-lane
N_LANES = 4
N_SIGNAL_LANES = 5   # audio_mod tracks: envelope, onsets, low, mid, high

#: Snap choices shown in the toolbar: (label, kind, value).
#: kind "beats" → value in beats, "seconds" → value in seconds, "off".
SNAP_CHOICES = (
    ("Bar", "beats", 4.0),
    ("Beat", "beats", 1.0),
    ("1/2 beat", "beats", 0.5),
    ("1/4 beat", "beats", 0.25),
    ("1 s", "seconds", 1.0),
    ("0.5 s", "seconds", 0.5),
    ("0.1 s", "seconds", 0.1),
    ("Pulse (25 ms)", "seconds", 0.025),
    ("Off", "off", 0.0),
)


@dataclass
class Snap:
    kind: str = "beats"
    value: float = 1.0

    def step_s(self, bpm: float) -> float:
        if self.kind == "beats":
            return self.value * 60.0 / max(1e-6, bpm)
        if self.kind == "seconds":
            return self.value
        return 0.0

    def apply(self, t: float, bpm: float, magnets=(), radius_s: float = 0.0) -> float:
        """Snap *t* to the grid; a magnet (marker) within *radius_s* wins."""
        m = nearest_magnet(t, magnets, radius_s)
        if m is not None:
            return m
        step = self.step_s(bpm)
        if step <= 0:
            return round(t * 40) / 40  # always land on a pulse
        return round(t / step) * step


def nearest_magnet(t: float, magnets, radius_s: float) -> float | None:
    """The magnet closest to *t* if it lies within *radius_s*, else None."""
    if radius_s <= 0:
        return None
    best = min(magnets, key=lambda m: abs(m - t), default=None)
    return best if best is not None and abs(best - t) <= radius_s else None


@dataclass(frozen=True)
class AutoLane:
    """An automated effect parameter shown as an extra lane under a track."""

    fx_index: int
    key: str
    label: str


@dataclass
class Row:
    track_id: str
    index: int
    y: float
    expanded: bool
    lanes: bool          # a standard track: four pulse lanes, takes generator clips
    auto_lanes: tuple[AutoLane, ...] = ()
    kind: str = "standard"

    @property
    def n_lanes(self) -> int:
        """Lanes shown below the clip row: 4 pulse lanes + automation lanes on a
        standard track, the 5 analysis signals on an audio_mod track."""
        if not self.expanded:
            return 0
        if self.kind == "audio_mod":
            return N_SIGNAL_LANES
        if self.kind == "video":
            return 1                  # the video's sound (waveform)
        if self.kind not in ("standard", "triphase"):
            return 0
        return N_LANES + len(self.auto_lanes)

    @property
    def takes_clips(self) -> bool:
        """Generator / pulse clips can be placed here (standard or triphase)."""
        return self.kind in ("standard", "triphase")

    @property
    def height(self) -> float:
        return CLIP_ROW_H + self.n_lanes * LANE_H

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def lane_y(self, lane: int) -> float:
        return self.y + CLIP_ROW_H + lane * LANE_H


def automation_lanes(track) -> tuple[AutoLane, ...]:
    """Automation lanes of a track: one per automated parameter of its effects."""
    from ...effects import get_effect
    out = []
    for i, eff in enumerate(track.effects):
        try:
            definition = get_effect(eff.get("type"))
        except KeyError:
            continue
        for key in (eff.get("automation") or {}):
            if key in definition.automatable:
                spec = definition.spec(key)
                out.append(AutoLane(i, key, f"{definition.label}: {spec.label}"))
    return tuple(out)


class SongLayout:
    def __init__(self, project: Project, expanded: set[str]):
        self.rows: list[Row] = []
        y = 0.0
        for i, track in enumerate(project.tracks):
            row = Row(track.id, i, y, track.id in expanded, track.type == "standard",
                      automation_lanes(track), track.type)
            self.rows.append(row)
            y = row.bottom
        self.height = y
        self._by_id = {r.track_id: r for r in self.rows}

    def row(self, track_id: str) -> Row | None:
        return self._by_id.get(track_id)

    def row_at(self, y: float, clamp: bool = False) -> Row | None:
        for r in self.rows:
            if r.y <= y < r.bottom:
                return r
        if clamp and self.rows:
            return self.rows[0] if y < 0 else self.rows[-1]
        return None


# ── adaptive grid / tick spacing ────────────────────────────────────────────
TIME_STEPS = (0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900,
              1800, 3600)


def time_step_for(pps: float, min_px: float) -> float:
    """Smallest 'round' time step that is at least *min_px* wide."""
    for step in TIME_STEPS:
        if step * pps >= min_px:
            return step
    return TIME_STEPS[-1] * math.ceil(min_px / (TIME_STEPS[-1] * pps))

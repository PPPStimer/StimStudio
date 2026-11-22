"""Funscript loading + resampling to pulse channels (ported from HWL Viewer).

Supports single-axis files, multi-axis file sets (``name.roll.funscript`` ...)
and single-file multi-axis containers (top-level ``axes`` array).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Canonical axis ids (TCode) with the filename suffixes / aliases used in the wild.
AXIS_ALIASES: dict[str, str] = {
    # linear
    "l0": "L0", "stroke": "L0", "": "L0", "raw": "L0", "up": "L0", "pos": "L0",
    "l1": "L1", "surge": "L1", "forward": "L1",
    "l2": "L2", "sway": "L2", "left": "L2",
    # rotation
    "r0": "R0", "twist": "R0", "yaw": "R0",
    "r1": "R1", "roll": "R1",
    "r2": "R2", "pitch": "R2",
    # auxiliary
    "v0": "V0", "vib": "V0", "vibrate": "V0",
    "v1": "V1", "pump": "V1", "lube": "V1",
    "v2": "V2", "valve": "V2", "suck": "V2",
    "a0": "A0", "a1": "A1", "a2": "A2",
}

AXIS_LABELS: dict[str, str] = {
    "L0": "L0 stroke",
    "L1": "L1 surge",
    "L2": "L2 sway",
    "R0": "R0 twist",
    "R1": "R1 roll",
    "R2": "R2 pitch",
    "V0": "V0 vibrate",
    "V1": "V1 pump",
    "V2": "V2 valve",
}

INTERP_LINEAR = "linear"
INTERP_HOLD   = "hold"
INTERP_SMOOTH = "smooth"

VALUE_POSITION = "position"
VALUE_SPEED    = "speed"

#: Motion speed (funscript units per second) that maps to 1.0 in speed mode.
DEFAULT_SPEED_REF = 400.0


def _canonical_axis(raw: str) -> str:
    key = (raw or "").strip().lower().lstrip(".")
    return AXIS_ALIASES.get(key, raw.strip() or "L0")


def axis_label(axis_id: str) -> str:
    return AXIS_LABELS.get(axis_id, axis_id)


@dataclass
class FunscriptAxis:
    """One position track extracted from a funscript file."""

    file_path: str
    file_name: str
    axis_id:   str
    times_ms:  np.ndarray          # int64, strictly increasing
    positions: np.ndarray          # float64, already normalised to 0..1
    inverted:  bool = False
    metadata:  dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.file_name} : {axis_label(self.axis_id)}"

    @property
    def n_actions(self) -> int:
        return int(self.times_ms.size)

    @property
    def last_ms(self) -> float:
        return float(self.times_ms[-1]) if self.times_ms.size else 0.0

    @property
    def meta_duration_ms(self) -> float:
        try:
            return float(self.metadata.get("duration") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def duration_ms(self, use_metadata: bool = False) -> float:
        d = self.last_ms
        if use_metadata:
            d = max(d, self.meta_duration_ms)
        return d


def _suffix_axis(path: Path) -> str:
    """``clip.roll.funscript`` -> ``R1``; ``clip.funscript`` -> ``L0``."""
    stem = path.stem                       # "clip.roll"
    if "." in stem:
        return _canonical_axis(stem.rsplit(".", 1)[1])
    return "L0"


def _parse_actions(raw_actions, range_pct, inverted: bool) -> tuple[np.ndarray, np.ndarray]:
    """Normalise an ``actions`` array into (times_ms, positions 0..1)."""
    pairs: list[tuple[int, float]] = []
    for a in raw_actions or []:
        if not isinstance(a, dict):
            continue
        at  = a.get("at",  a.get("t"))
        pos = a.get("pos", a.get("value"))
        if at is None or pos is None:
            continue
        try:
            pairs.append((int(round(float(at))), float(pos)))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    pairs.sort(key=lambda p: p[0])
    times = np.fromiter((p[0] for p in pairs), dtype=np.int64,   count=len(pairs))
    pos   = np.fromiter((p[1] for p in pairs), dtype=np.float64, count=len(pairs))

    # Drop duplicate timestamps, keeping the last value at each instant.
    keep = np.ones(times.size, dtype=bool)
    keep[:-1] = times[1:] != times[:-1]
    times, pos = times[keep], pos[keep]

    # `range` is a percentage cap the script was authored against (0 = unset).
    try:
        rng = float(range_pct)
    except (TypeError, ValueError):
        rng = 0.0
    scale = rng if 0.0 < rng <= 100.0 else 100.0
    pos = np.clip(pos / scale, 0.0, 1.0)
    if inverted:
        pos = 1.0 - pos
    return times, pos


def load_funscript_axes(path: str | Path) -> list[FunscriptAxis]:
    """Load every axis contained in *path*. Raises ValueError on bad input."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p.name}: not valid JSON ({exc.msg} at line {exc.lineno})") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{p.name}: expected a JSON object at the top level")

    metadata  = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    top_range = doc.get("range", 0)
    top_inv   = bool(doc.get("inverted", False))
    out: list[FunscriptAxis] = []

    def add(axis_id: str, actions, rng, inv) -> None:
        times, pos = _parse_actions(actions, rng, inv)
        if times.size == 0:
            return
        out.append(FunscriptAxis(
            file_path=str(p), file_name=p.name, axis_id=axis_id,
            times_ms=times, positions=pos, inverted=bool(inv), metadata=metadata))

    if isinstance(doc.get("actions"), list):
        add(_suffix_axis(p), doc["actions"], top_range, top_inv)

    # Multi-axis container: {"axes": [{"id": "R1", "actions": [...]}, ...]}
    for entry in doc.get("axes") or []:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("id") or entry.get("name") or entry.get("axis") or ""
        add(_canonical_axis(str(raw_id)),
            entry.get("actions"),
            entry.get("range", top_range),
            entry.get("inverted", top_inv))

    if not out:
        raise ValueError(f"{p.name}: no actions found")

    # Deduplicate axis ids within one file (e.g. two "L0" tracks).
    seen: dict[str, int] = {}
    for ax in out:
        n = seen.get(ax.axis_id, 0)
        seen[ax.axis_id] = n + 1
        if n:
            ax.axis_id = f"{ax.axis_id}#{n + 1}"
    return out


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def sample_positions(axis: FunscriptAxis, grid_ms: np.ndarray,
                     interp: str = INTERP_LINEAR) -> np.ndarray:
    """Sample the axis position (0..1) on *grid_ms*, held flat outside its range."""
    t = axis.times_ms.astype(np.float64)
    v = axis.positions
    if t.size == 1:
        return np.full(grid_ms.shape, v[0], dtype=np.float64)

    if interp == INTERP_HOLD:
        idx = np.searchsorted(t, grid_ms, side="right") - 1
        return v[np.clip(idx, 0, v.size - 1)]

    if interp == INTERP_SMOOTH:
        idx    = np.clip(np.searchsorted(t, grid_ms, side="right") - 1, 0, t.size - 2)
        t0, t1 = t[idx], t[idx + 1]
        span   = np.where(t1 > t0, t1 - t0, 1.0)
        frac   = np.clip((grid_ms - t0) / span, 0.0, 1.0)
        return v[idx] + (v[idx + 1] - v[idx]) * _smoothstep(frac)

    return np.interp(grid_ms, t, v)


def sample_speed(axis: FunscriptAxis, grid_ms: np.ndarray,
                 speed_ref: float = DEFAULT_SPEED_REF,
                 interp: str = INTERP_LINEAR) -> np.ndarray:
    """Sample |motion speed| normalised so *speed_ref* units/s maps to 1.0.

    Speed is constant across each action interval (the usual funscript metric);
    every mode except ``INTERP_HOLD`` softens the steps with a small kernel.
    """
    t = axis.times_ms.astype(np.float64)
    v = axis.positions
    if t.size < 2:
        return np.zeros(grid_ms.shape, dtype=np.float64)

    dt  = np.maximum(t[1:] - t[:-1], 1.0)                # ms, guard against 0
    seg = np.abs(v[1:] - v[:-1]) * 100.0 / dt * 1000.0   # funscript units / s
    ref = speed_ref if speed_ref > 0 else DEFAULT_SPEED_REF
    seg = np.clip(seg / ref, 0.0, 1.0)

    idx = np.clip(np.searchsorted(t, grid_ms, side="right") - 1, 0, seg.size - 1)
    out = seg[idx]
    if interp != INTERP_HOLD and out.size >= 3:
        smoothed     = np.convolve(out, np.array([0.25, 0.5, 0.25]), mode="same")
        smoothed[0]  = out[0]
        smoothed[-1] = out[-1]
        out = smoothed
    # Silence outside the scripted range.
    out[(grid_ms < t[0]) | (grid_ms > t[-1])] = 0.0
    return out


def render_channel(axis: FunscriptAxis | None, grid_ms: np.ndarray, *,
                   value: str = VALUE_POSITION,
                   interp: str = INTERP_LINEAR,
                   invert: bool = False,
                   out_min: float = 0.0,
                   out_max: float = 1.0,
                   speed_ref: float = DEFAULT_SPEED_REF) -> np.ndarray:
    """Render one HWL channel from an axis (or a constant when *axis* is None)."""
    if axis is None:
        return np.full(grid_ms.shape, float(np.clip(out_min, 0.0, 1.0)), dtype=np.float64)

    if value == VALUE_SPEED:
        raw = sample_speed(axis, grid_ms, speed_ref, interp)
    else:
        raw = sample_positions(axis, grid_ms, interp)

    if invert:
        raw = 1.0 - raw
    return np.clip(out_min + raw * (out_max - out_min), 0.0, 1.0)


def pulse_grid(n_pulses: int, fs: int, offset_ms: float = 0.0) -> np.ndarray:
    """Timestamps (ms) of *n_pulses* pulses at *fs* Hz, shifted by *offset_ms*."""
    return np.arange(n_pulses, dtype=np.float64) * (1000.0 / fs) - offset_ms


# ── StimStudio: axes → pulse columns ───────────────────────────────────────
@dataclass
class ChannelMap:
    """How one output column is rendered (see :func:`render_channel`)."""

    axis: int | None = None          # index into the loaded axes; None = constant out_min
    value: str = VALUE_POSITION
    invert: bool = False
    out_min: float = 0.0
    out_max: float = 1.0


def auto_mapping(axes: list[FunscriptAxis]) -> list[ChannelMap]:
    """HWL Viewer's default: axis 0 → both amplitudes (or 0 / 1 when there are two),
    axes 2 / 3 → frequencies; frequency columns without an axis sit at 0.5."""
    picks = [0, 1 if len(axes) > 1 else 0, 2, 3]
    out = []
    for col, idx in enumerate(picks):
        if idx < len(axes):
            out.append(ChannelMap(axis=idx))
        else:
            out.append(ChannelMap(axis=None, out_min=0.5 if col >= 2 else 0.0))
    return out


def axes_duration_ms(axes: list[FunscriptAxis], use_metadata: bool = True) -> float:
    return max((ax.duration_ms(use_metadata) for ax in axes), default=0.0)


def funscript_to_pulses(axes: list[FunscriptAxis], mapping: list[ChannelMap], fs: int = 40,
                        interp: str = INTERP_LINEAR, offset_ms: float = 0.0,
                        speed_ref: float = DEFAULT_SPEED_REF,
                        duration_ms: float | None = None) -> np.ndarray:
    """Render four columns from funscript axes into an (N, 4) float32 pulse array."""
    if duration_ms is None:
        duration_ms = axes_duration_ms(axes) + max(0.0, offset_ms)
    n = max(1, int(np.ceil(duration_ms / 1000.0 * fs)))
    grid = pulse_grid(n, fs, offset_ms)
    out = np.zeros((n, 4), dtype=np.float32)
    for col, m in enumerate(mapping[:4]):
        axis = axes[m.axis] if m.axis is not None and 0 <= m.axis < len(axes) else None
        out[:, col] = render_channel(axis, grid, value=m.value, interp=interp, invert=m.invert,
                                     out_min=m.out_min, out_max=m.out_max, speed_ref=speed_ref)
    return out

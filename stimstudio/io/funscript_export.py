"""Funscript export: the four stim lanes (amplitude and frequency, A and B) as actions.

Each lane (0..1 at 40 Hz) becomes positions 0..100. Points that lie on the
straight line between their neighbours (within *tolerance* position units)
are dropped — Ramer–Douglas–Peucker on the vertical distance — so a file
keeps the shape with a small fraction of the 40 actions per second.

Two layouts:

* ``separate`` — ``name.amp_a.funscript``, ``name.amp_b.funscript``,
  ``name.freq_a.funscript``, ``name.freq_b.funscript`` (one axis per file,
  opened by any player);
* ``single`` — one ``name.funscript`` whose ``axes`` array holds all four.

:func:`stimstudio.io.funscript.auto_mapping` recognises these axis ids, so an
exported set imports straight back onto the right lanes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.pulse import FS

#: axis id per pulse column (amp L/A, amp R/B, freq L/A, freq R/B)
AXIS_IDS = ("amp_a", "amp_b", "freq_a", "freq_b")
AXIS_TITLES = {"amp_a": "Amplitude, channel A", "amp_b": "Amplitude, channel B",
               "freq_a": "Pulse frequency, channel A (0..100 = the frequency range)",
               "freq_b": "Pulse frequency, channel B (0..100 = the frequency range)"}
LAYOUT_SEPARATE = "separate"
LAYOUT_SINGLE = "single"
DEFAULT_TOLERANCE = 1.0


def simplify(times_ms: np.ndarray, pos: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices of the points to keep (sorted), endpoints always included.

    Iterative Ramer–Douglas–Peucker with the vertical distance to the chord.
    With ``tolerance <= 0`` only the inner points of flat runs are removed.
    """
    n = len(pos)
    if n <= 2:
        return np.arange(n)
    t = np.asarray(times_ms, dtype=np.float64)
    v = np.asarray(pos, dtype=np.float64)
    if tolerance <= 0:
        same = (v[1:-1] == v[:-2]) & (v[1:-1] == v[2:])
        keep = np.ones(n, dtype=bool)
        keep[1:-1] = ~same
        return np.flatnonzero(keep)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg_t = t[a + 1:b]
        chord = v[a] + (v[b] - v[a]) * (seg_t - t[a]) / (t[b] - t[a])
        dist = np.abs(v[a + 1:b] - chord)
        i = int(np.argmax(dist))
        if dist[i] > tolerance:
            k = a + 1 + i
            keep[k] = True
            stack.append((a, k))
            stack.append((k, b))
    return np.flatnonzero(keep)


def lane_to_actions(values: np.ndarray, tolerance: float = DEFAULT_TOLERANCE,
                    fs: int = FS) -> list[dict]:
    """One lane (0..1 per pulse) → ``[{"at": ms, "pos": 0..100}, …]``, simplified."""
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if not len(values):
        return []
    pos = np.rint(values * 100.0)
    times = np.rint(np.arange(len(values)) * (1000.0 / fs))
    idx = simplify(times, pos, tolerance)
    return [{"at": int(times[i]), "pos": int(pos[i])} for i in idx]


def _metadata(title: str, duration_ms: int, description: str, extra: dict | None) -> dict:
    from .. import __version__
    meta = {"creator": f"StimStudio {__version__}", "title": title,
            "description": description, "duration": int(duration_ms),
            "type": "basic", "license": "", "notes": "", "performers": [], "script_url": "",
            "tags": ["estim"], "video_url": ""}
    meta.update(extra or {})
    return meta


def base_path(path) -> Path:
    """``x.funscript`` / ``x.amp_a.funscript`` / ``x`` → ``x`` (without any suffix)."""
    p = Path(path)
    name = p.name
    if name.lower().endswith(".funscript"):
        name = name[: -len(".funscript")]
    for axis in AXIS_IDS:
        if name.lower().endswith("." + axis):
            name = name[: -len(axis) - 1]
    return p.with_name(name)


def output_paths(path, layout: str = LAYOUT_SEPARATE) -> list[Path]:
    base = base_path(path)
    if layout == LAYOUT_SINGLE:
        return [base.with_name(base.name + ".funscript")]
    return [base.with_name(f"{base.name}.{axis}.funscript") for axis in AXIS_IDS]


def write_funscripts(path, pulses: np.ndarray, layout: str = LAYOUT_SEPARATE,
                     tolerance: float = DEFAULT_TOLERANCE, metadata: dict | None = None,
                     fs: int = FS) -> list[Path]:
    """Write the four lanes of *pulses* (N, 4); returns the files written."""
    pulses = np.asarray(pulses)
    if pulses.ndim != 2 or pulses.shape[1] < 4 or not len(pulses):
        raise ValueError("nothing to export")
    duration_ms = int(round(len(pulses) * 1000.0 / fs))
    base = base_path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    actions = {axis: lane_to_actions(pulses[:, col], tolerance, fs)
               for col, axis in enumerate(AXIS_IDS)}
    paths = output_paths(path, layout)
    if layout == LAYOUT_SINGLE:
        doc = {"version": "1.0", "inverted": False, "range": 100,
               "metadata": _metadata(base.name, duration_ms,
                                     "E-stim lanes: amplitude and pulse frequency, "
                                     "channels A and B", metadata),
               "actions": [],
               "axes": [{"id": axis, "actions": actions[axis]} for axis in AXIS_IDS]}
        _dump(paths[0], doc)
        return paths
    for axis, p in zip(AXIS_IDS, paths):
        doc = {"version": "1.0", "inverted": False, "range": 100,
               "metadata": _metadata(f"{base.name} — {axis}", duration_ms,
                                     AXIS_TITLES[axis], metadata),
               "actions": actions[axis]}
        _dump(p, doc)
    return paths


def _dump(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")

"""Exports: .hwl and audio, whole project or a time range, mix or per-track stems.

Qt-free: used by the GUI export dialog and the command line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import Project
from .pulse import FS
from .render import ProjectRender, RenderCache, render_project

HWL_FORMAT = "hwl"
AUDIO_FORMATS = ("wav", "flac")
AUDIO_MODES = ("continuous", "wavelet")


@dataclass
class AudioSettings:
    mode: str = "continuous"
    sample_rate: int = 48000
    freq_hz: tuple[float, float] | None = None   # None = the mode's default range
    device_settings: dict | None = None          # shape, carrier, wavelet width / fade
    power: tuple[int, int] = (200, 200)


def slice_range(n: int, start_s: float | None, end_s: float | None) -> tuple[int, int]:
    a = 0 if start_s is None else max(0, round(start_s * FS))
    b = n if end_s is None else min(n, round(end_s * FS))
    if b <= a:
        raise ValueError("the export range is empty")
    return a, b


def mix_pulses(render: ProjectRender, a: int, b: int) -> np.ndarray:
    return render.master[a:b]


def mix_audio(render: ProjectRender, a: int, b: int, audio: AudioSettings,
              cal: dict | None = None) -> np.ndarray:
    """Audio of a render range. Continuous mode plays triphase with real phase;
    wavelet uses the A/B mix (as the live outputs do)."""
    from ..outputs.audio import render_offline
    settings = {**(audio.device_settings or {}), "triphase_cal": dict(cal or {})}
    if audio.mode == "continuous" and render.triphase is not None:
        return render_offline(render.master_audio[a:b], "continuous", audio.sample_rate,
                              settings, audio.freq_hz, audio.power,
                              triphase=render.triphase[a:b])
    return render_offline(render.master[a:b], audio.mode, audio.sample_rate, settings,
                          audio.freq_hz, audio.power)


def write_output(path, project: Project, render: ProjectRender, a: int, b: int,
                 audio: AudioSettings | None = None) -> int:
    """Write one file; the format follows the extension. Returns pulses / samples written."""
    from ..io.hwl import write_hwl
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext == HWL_FORMAT:
        pulses = mix_pulses(render, a, b)
        write_hwl(path, pulses)
        return len(pulses)
    if ext in AUDIO_FORMATS:
        import soundfile as sf
        audio = audio or AudioSettings()
        data = mix_audio(render, a, b, audio, project.triphase_cal)
        sf.write(str(path), data, audio.sample_rate, subtype="PCM_24")
        return len(data)
    raise ValueError(f"unsupported export format: .{ext}")


def export_mix(project: Project, path, start_s=None, end_s=None,
               audio: AudioSettings | None = None, cache: RenderCache | None = None) -> int:
    render = render_project(project, cache)
    a, b = slice_range(render.n, start_s, end_s)
    return write_output(path, project, render, a, b, audio)


def stem_tracks(project: Project) -> list:
    """Tracks that get a stem: every non-muted standard / triphase track."""
    return [t for t in project.tracks if t.type in ("standard", "triphase") and not t.muted]


def stem_project(project: Project, track_id: str) -> Project:
    """A copy of *project* in which only *track_id* is heard (modulators keep working,
    master effects stay on)."""
    p = project.clone()
    for t in p.tracks:
        t.solo = t.id == track_id
        if t.id == track_id:
            t.muted = False
    if p.length_s is None:     # keep the full project length in every stem
        p.length_s = project.duration_s()
    return p


def safe_name(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_()" else "_" for c in name).strip()
    return keep or "track"


def export_stems(project: Project, folder, base: str, fmt: str = HWL_FORMAT,
                 start_s=None, end_s=None, audio: AudioSettings | None = None,
                 cache: RenderCache | None = None) -> list[Path]:
    """One file per track: ``<base> - NN <track>.<fmt>``. Returns the paths written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for i, track in enumerate(stem_tracks(project), start=1):
        path = folder / f"{base} - {i:02d} {safe_name(track.name)}.{fmt}"
        render = render_project(stem_project(project, track.id), cache)
        a, b = slice_range(render.n, start_s, end_s)
        write_output(path, project, render, a, b, audio)
        written.append(path)
    return written

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
FUNSCRIPT_FORMAT = "funscript"
AUDIO_FORMATS = ("wav", "flac")
AUDIO_MODES = ("continuous", "wavelet")
AUDIO_SUBTYPE = "PCM_24"
AUDIO_BYTES_PER_FRAME = 2 * 3          # stereo, 24-bit
WAV_HEADER_BYTES = 44


@dataclass
class AudioSettings:
    mode: str = "continuous"
    sample_rate: int = 48000
    freq_hz: tuple[float, float] | None = None   # None = the mode's default range
    device_settings: dict | None = None          # shape, carrier, wavelet width / fade
    power: tuple[int, int] = (200, 200)


@dataclass
class FunscriptSettings:
    layout: str = "separate"          # "separate" (4 files) | "single" (one multi-axis file)
    tolerance: float = 1.0            # simplification, in position units (0..100)


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
                 audio: AudioSettings | None = None,
                 funscript: FunscriptSettings | None = None) -> int:
    """Write one file (a funscript set may be several); the format follows the
    extension. Returns pulses / samples written."""
    from ..io.hwl import write_hwl
    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext == HWL_FORMAT:
        pulses = mix_pulses(render, a, b)
        write_hwl(path, pulses)
        return len(pulses)
    if ext == FUNSCRIPT_FORMAT:
        from ..io.funscript_export import write_funscripts
        fs = funscript or FunscriptSettings()
        pulses = mix_pulses(render, a, b)
        write_funscripts(path, pulses, fs.layout, fs.tolerance)
        return len(pulses)
    if ext in AUDIO_FORMATS:
        import soundfile as sf
        audio = audio or AudioSettings()
        data = mix_audio(render, a, b, audio, project.triphase_cal)
        sf.write(str(path), data, audio.sample_rate, subtype=AUDIO_SUBTYPE)
        return len(data)
    raise ValueError(f"unsupported export format: .{ext}")


def written_paths(path, fmt: str, funscript: FunscriptSettings | None = None) -> list[Path]:
    """The files :func:`write_output` produces for *path* (a funscript set may be four)."""
    if fmt == FUNSCRIPT_FORMAT:
        from ..io.funscript_export import output_paths
        return output_paths(path, (funscript or FunscriptSettings()).layout)
    return [Path(path)]


def export_mix(project: Project, path, start_s=None, end_s=None,
               audio: AudioSettings | None = None, cache: RenderCache | None = None,
               funscript: FunscriptSettings | None = None) -> int:
    render = render_project(project, cache)
    a, b = slice_range(render.n, start_s, end_s)
    return write_output(path, project, render, a, b, audio, funscript)


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
                 cache: RenderCache | None = None,
                 funscript: FunscriptSettings | None = None) -> list[Path]:
    """One file per track: ``<base> - NN <track>.<fmt>``. Returns the paths written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for i, track in enumerate(stem_tracks(project), start=1):
        path = folder / f"{base} - {i:02d} {safe_name(track.name)}.{fmt}"
        render = render_project(stem_project(project, track.id), cache)
        a, b = slice_range(render.n, start_s, end_s)
        write_output(path, project, render, a, b, audio, funscript)
        written.extend(written_paths(path, fmt, funscript))
    return written


# ── what an audio export will contain ──────────────────────────────────────
def _size_text(n_bytes: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.0f} {unit}" if unit == "bytes" else f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return ""


def _clock(seconds: float) -> str:
    minutes, sec = divmod(max(0.0, seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}:{minutes:02d}:{sec:06.3f}" if hours else f"{minutes:02d}:{sec:06.3f}"


def describe_audio(project: Project, n_pulses: int, start_s=None, end_s=None,
                   audio: AudioSettings | None = None, fmt: str = "wav", stems: int = 0,
                   from_panel: bool = False) -> list[tuple[str, str]]:
    """Human-readable parameters of the audio an export would write, as (label, value).

    *n_pulses* is the project's rendered length; the range is clipped to it the
    same way :func:`slice_range` does. Nothing is rendered here.
    """
    from ..outputs.audio import AUDIO_MODES as SINKS
    audio = audio or AudioSettings()
    sr = int(audio.sample_rate)
    try:
        a, b = slice_range(n_pulses, start_s, end_s)
    except ValueError:
        a = b = 0
    seconds = (b - a) / FS
    frames = int(np.ceil(seconds * sr))
    files = max(1, stems)
    rows: list[tuple[str, str]] = []
    rows.append(("File", f"{fmt.upper()}, PCM 24-bit, stereo — left = channel A, "
                         f"right = channel B" + (f", {stems} stem files" if stems else "")))
    rows.append(("Sample rate", f"{sr} Hz"))
    rows.append(("Duration", f"{_clock(seconds)}  ({_clock(a / FS)} – {_clock(b / FS)}), "
                             f"{frames:,} samples".replace(",", " ")))
    wav_bytes = WAV_HEADER_BYTES + frames * AUDIO_BYTES_PER_FRAME
    if fmt == "wav":
        size = _size_text(wav_bytes * files)
    else:
        size = f"≤ {_size_text(wav_bytes * files)} (FLAC is lossless, usually much smaller)"
    rows.append(("Size", size + (" in total" if stems else "")))

    sink = SINKS.get(audio.mode)
    fmap = sink.default_freq_map if sink is not None else None
    if fmap is not None and audio.freq_hz is not None:
        fmap = fmap.with_hz(*sorted(audio.freq_hz))
    lo, hi = (fmap.min_hz, fmap.max_hz) if fmap is not None else (0, 0)
    dev = audio.device_settings or {}
    shape = dev.get("shape", "sine")
    if audio.mode == "wavelet":
        rows.append(("Mode", "Wavelet — short bursts of a fixed carrier, repeated at the "
                             "pulse frequency"))
        rows.append(("Carrier", f"{int(dev.get('carrier', 1000))} Hz, "
                                f"{int(dev.get('wavelet_width', 5))} cycles per burst, "
                                f"fade {float(dev.get('wavelet_fade', 0.5)):.2f}"))
        rows.append(("Burst rate", f"{lo}–{hi} Hz (follows the frequency lanes)"))
    else:
        rows.append(("Mode", "Continuous — an unbroken tone per channel"))
        rows.append(("Tone pitch", f"{lo}–{hi} Hz (follows the frequency lanes)"))
    rows.append(("Wave shape", shape))
    tri = any(t.type == "triphase" and not t.muted for t in project.tracks)
    if not tri:
        tri_text = "none in the project"
    elif audio.mode == "continuous":
        tri_text = "phase-true three-phase signal (both channels share one carrier)"
    else:
        tri_text = "mixed into channels A / B (wavelet has no phase)"
    rows.append(("Triphase", tri_text))
    pa, pb = audio.power
    rows.append(("Level", f"full scale (power {pa}/{pb}) — the Output panel's power, mute "
                          "and soft start are not applied; peaks clipped to ±1"))
    rows.append(("Settings from", "the Output panel" if from_panel else
                 f"the defaults of the {audio.mode} mode"))
    return rows

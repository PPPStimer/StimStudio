"""WAV/audio import: derive a normalized PulseTrack from an audio file.

Amplitude comes from a windowed RMS envelope; frequency from per-window
zero-crossing rate mapped into [floor_hz, ceil_hz] -> [0, 1]. Stereo files
map L/R directly; mono duplicates across channels. Re-rendering reads the
referenced file again, so the path must remain valid for the sequencer.
"""

from __future__ import annotations

import numpy as np

from ..core.pulse import FS, PulseTrack
from .base import Generator, ParamSpec, register


def _audio_duration_s(path: str) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def _envelope_and_pitch(samples: np.ndarray, sr: int, n: int, floor_hz: float, ceil_hz: float):
    """Return (amp[0..1], freq[0..1]) of length n for a 1-D sample array."""
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    edges = np.linspace(0, len(samples), n + 1).astype(int)
    amp = np.zeros(n, dtype=np.float32)
    freq = np.zeros(n, dtype=np.float32)
    span = max(1e-6, ceil_hz - floor_hz)
    for i in range(n):
        seg = samples[edges[i]:edges[i + 1]]
        if seg.size == 0:
            continue
        amp[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2))
        crossings = np.count_nonzero(np.diff(np.signbit(seg)))
        win_s = seg.size / sr
        hz = (crossings / 2.0) / win_s if win_s > 0 else 0.0
        freq[i] = np.clip((hz - floor_hz) / span, 0.0, 1.0)
    peak = amp.max()
    if peak > 0:
        amp /= peak
    return amp, freq


@register
class WavImportGenerator(Generator):
    type_id = "wav"
    label = "WAV import"
    category = "import"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("path", "Audio file", "path", ""),
            ParamSpec("gain", "Amp gain", "float", 1.0, 0.0, 4.0, 0.05),
            ParamSpec("floor_hz", "Freq floor (Hz)", "float", 0.0, 0.0, 2000.0, 10.0),
            ParamSpec("ceil_hz", "Freq ceil (Hz)", "float", 500.0, 10.0, 4000.0, 10.0),
        ]

    @staticmethod
    def file_duration_s(path: str) -> float:
        return _audio_duration_s(path)

    def render(self, n: int, params: dict, seed: int = 0) -> PulseTrack:
        import soundfile as sf

        p = self.merged_params(params)
        path = str(p.get("path", "")).strip()
        if not path:
            return PulseTrack.zeros(n)
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        floor_hz, ceil_hz = float(p["floor_hz"]), float(p["ceil_hz"])
        gain = float(p["gain"])
        left = data[:, 0]
        right = data[:, 1] if data.shape[1] > 1 else data[:, 0]
        l_amp, l_freq = _envelope_and_pitch(left, sr, n, floor_hz, ceil_hz)
        r_amp, r_freq = _envelope_and_pitch(right, sr, n, floor_hz, ceil_hz)
        return PulseTrack(
            np.clip(l_amp * gain, 0, 1),
            np.clip(r_amp * gain, 0, 1),
            l_freq,
            r_freq,
        )

"""Basic waveform generator: shaped amplitude/frequency envelopes per channel."""

from __future__ import annotations

import numpy as np

from ..core.pulse import FS, PulseTrack
from ..core.shapes import SHAPES, shape_wave  # noqa: F401  (re-exported)
from .base import Generator, ParamSpec, register


@register
class WaveformGenerator(Generator):
    type_id = "waveform"
    label = "Waveform"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("amp_shape", "Amp shape", "choice", "sine", choices=SHAPES),
            ParamSpec("amp_speed", "Amp speed (Hz)", "float", 0.5, 0.0, 20.0, 0.05),
            ParamSpec("amp_min", "Amp min", "float", 0.0, 0.0, 1.0, 0.01),
            ParamSpec("amp_max", "Amp max", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("freq_shape", "Freq shape", "choice", "sine", choices=SHAPES),
            ParamSpec("freq_speed", "Freq speed (Hz)", "float", 0.2, 0.0, 20.0, 0.05),
            ParamSpec("freq_min", "Freq min", "float", 0.2, 0.0, 1.0, 0.01),
            ParamSpec("freq_max", "Freq max", "float", 0.8, 0.0, 1.0, 0.01),
            ParamSpec("duty", "Pulse duty", "float", 0.5, 0.01, 0.99, 0.01),
            ParamSpec("phase", "Phase", "float", 0.0, 0.0, 1.0, 0.01),
            ParamSpec("channel_phase", "A/B phase offset", "float", 0.0, 0.0, 1.0, 0.01),
        ]

    def render(self, n: int, params: dict, seed: int = 0) -> PulseTrack:
        p = self.merged_params(params)
        t = np.arange(n, dtype=np.float32) / FS
        duty = float(p["duty"])
        phase = float(p["phase"])
        ch = float(p["channel_phase"])

        def env(shape, speed, lo, hi, extra_phase):
            cyc = t * float(speed) + phase + extra_phase
            return float(lo) + (float(hi) - float(lo)) * shape_wave(shape, cyc, duty)

        l_amp = env(p["amp_shape"], p["amp_speed"], p["amp_min"], p["amp_max"], 0.0)
        r_amp = env(p["amp_shape"], p["amp_speed"], p["amp_min"], p["amp_max"], ch)
        l_freq = env(p["freq_shape"], p["freq_speed"], p["freq_min"], p["freq_max"], 0.0)
        r_freq = env(p["freq_shape"], p["freq_speed"], p["freq_min"], p["freq_max"], ch)
        return PulseTrack(l_amp, r_amp, l_freq, r_freq)

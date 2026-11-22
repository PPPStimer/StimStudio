"""ADSR amplitude envelope generator."""
from __future__ import annotations

import numpy as np

from ..core.pulse import FS, PulseTrack
from .base import Generator, ParamSpec, register
from .waveforms import SHAPES, shape_wave


@register
class ADSRGenerator(Generator):
    type_id = "adsr"
    label = "ADSR"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("attack_s",      "Attack",         "float", 1.0,  0.0,  30.0, 0.05),
            ParamSpec("decay_s",       "Decay",          "float", 0.5,  0.0,  30.0, 0.05),
            ParamSpec("sustain",       "Sustain level",  "float", 0.7,  0.0,  1.0,  0.01),
            ParamSpec("release_s",     "Release",        "float", 1.0,  0.0,  30.0, 0.05),
            ParamSpec("peak_amp",      "Peak amplitude", "float", 1.0,  0.0,  1.0,  0.01),
            ParamSpec("freq_shape",    "Freq shape",     "choice","sine", choices=SHAPES),
            ParamSpec("freq_speed",    "Freq speed (Hz)","float", 0.2,  0.0,  20.0, 0.05),
            ParamSpec("freq_min",      "Freq min",       "float", 0.2,  0.0,  1.0,  0.01),
            ParamSpec("freq_max",      "Freq max",       "float", 0.8,  0.0,  1.0,  0.01),
            ParamSpec("channel_phase", "A/B phase",      "float", 0.0,  0.0,  1.0,  0.01),
        ]

    def render(self, n: int, params: dict, seed: int = 0) -> PulseTrack:
        p = self.merged_params(params)
        t = np.arange(n, dtype=np.float32) / FS
        duration = n / FS

        attack  = float(p["attack_s"])
        decay   = float(p["decay_s"])
        sustain = float(p["sustain"])
        release = float(p["release_s"])
        peak    = float(p["peak_amp"])

        t_d = attack                             # decay phase start
        t_s = attack + decay                     # sustain phase start
        t_r = max(t_s, duration - release)       # release phase start

        amp = np.empty(n, dtype=np.float32)

        mask = t < t_d
        amp[mask] = (peak * t[mask] / attack) if attack > 0 else peak

        mask = (t >= t_d) & (t < t_s)
        if decay > 0:
            amp[mask] = peak + (sustain - peak) * ((t[mask] - t_d) / decay)
        else:
            amp[mask] = sustain

        mask = (t >= t_s) & (t < t_r)
        amp[mask] = sustain

        mask = t >= t_r
        if release > 0:
            amp[mask] = sustain * (1.0 - np.clip((t[mask] - t_r) / release, 0.0, 1.0))
        else:
            amp[mask] = 0.0

        ch = float(p["channel_phase"])
        fmin, fmax = float(p["freq_min"]), float(p["freq_max"])
        l_freq = (fmin + (fmax - fmin) * shape_wave(p["freq_shape"],
                                                     t * float(p["freq_speed"]))).astype(np.float32)
        r_freq = (fmin + (fmax - fmin) * shape_wave(p["freq_shape"],
                                                     t * float(p["freq_speed"]) + ch)).astype(np.float32)

        return PulseTrack(amp, amp.copy(), l_freq, r_freq)

"""Channel-motion generator: independent A/B with phase, ping-pong, panning."""

from __future__ import annotations

import numpy as np

from ..core.pulse import FS, PulseTrack
from .base import Generator, ParamSpec, register
from .waveforms import SHAPES, shape_wave


@register
class ChannelMotionGenerator(Generator):
    type_id = "channels"
    label = "Channels A/B"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("amp_shape", "Amp shape", "choice", "sine", choices=SHAPES),
            ParamSpec("amp_speed", "Amp speed (Hz)", "float", 0.8, 0.0, 20.0, 0.05),
            ParamSpec("amp_min", "Amp min", "float", 0.1, 0.0, 1.0, 0.01),
            ParamSpec("amp_max", "Amp max", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("freq", "Frequency", "float", 0.5, 0.0, 1.0, 0.01),
            ParamSpec("pan_speed", "Pan speed (Hz)", "float", 0.25, 0.0, 10.0, 0.05),
            ParamSpec("pan_depth", "Pan depth", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec(
                "mode", "Pan mode", "choice", "ping_pong",
                choices=("ping_pong", "phase", "static"),
            ),
            ParamSpec("phase_offset", "A/B phase", "float", 0.5, 0.0, 1.0, 0.01),
            ParamSpec("duty", "Pulse duty", "float", 0.5, 0.01, 0.99, 0.01),
        ]

    def render(self, n: int, params: dict, seed: int = 0) -> PulseTrack:
        p = self.merged_params(params)
        t = np.arange(n, dtype=np.float32) / FS
        duty = float(p["duty"])
        lo, hi = float(p["amp_min"]), float(p["amp_max"])
        mode = p["mode"]

        base = shape_wave(p["amp_shape"], t * float(p["amp_speed"]), duty)
        env = lo + (hi - lo) * base

        if mode == "phase":
            off = float(p["phase_offset"])
            l_amp = lo + (hi - lo) * shape_wave(p["amp_shape"], t * float(p["amp_speed"]), duty)
            r_amp = lo + (hi - lo) * shape_wave(
                p["amp_shape"], t * float(p["amp_speed"]) + off, duty
            )
        elif mode == "static":
            l_amp = env
            r_amp = env.copy()
        else:  # ping_pong: power slides between A and B
            pan = shape_wave("triangle", t * float(p["pan_speed"]))  # 0..1
            depth = float(p["pan_depth"])
            l_gain = 1.0 - depth * pan
            r_gain = 1.0 - depth * (1.0 - pan)
            l_amp = env * l_gain
            r_amp = env * r_gain

        freq = np.full(n, float(p["freq"]), dtype=np.float32)
        return PulseTrack(l_amp.astype(np.float32), r_amp.astype(np.float32), freq, freq.copy())

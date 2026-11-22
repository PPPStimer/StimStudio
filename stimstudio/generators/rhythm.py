"""Rhythm and build-up generators: Euclidean rhythm (tempo-synced) and Ramp / tease."""

from __future__ import annotations

import numpy as np

from ..core.pulse import FS, PulseTrack
from .base import Generator, ParamSpec, register

DIVISIONS = {"1/2": 2.0, "1/4": 1.0, "1/8": 0.5, "1/16": 0.25, "1/8T": 1.0 / 3.0,
             "1/16T": 1.0 / 6.0}
HIT_SHAPES = ("decay", "square", "sine", "ramp_up")


def euclid_pattern(hits: int, steps: int, rotate: int = 0) -> np.ndarray:
    """Boolean Euclidean rhythm: *hits* onsets spread as evenly as possible over
    *steps*, rotated left by *rotate*. E(3, 8) = x..x..x."""
    steps = max(1, int(steps))
    hits = max(0, min(steps, int(hits)))
    i = np.arange(steps)
    pattern = (i * hits) % steps < hits
    return np.roll(pattern, -int(rotate))


def _hit_envelope(shape: str, u: np.ndarray) -> np.ndarray:
    """Envelope of one hit for u in [0, 1) (position inside the hit)."""
    if shape == "square":
        return np.ones_like(u)
    if shape == "sine":
        return np.sin(np.pi * u)
    if shape == "ramp_up":
        return u
    return np.exp(-4.0 * u)  # decay


@register
class EuclidGenerator(Generator):
    type_id = "euclid"
    label = "Euclidean rhythm"
    category = "rhythm"
    tempo_synced = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("hits", "Hits", "int", 3, 0, 32),
            ParamSpec("steps", "Steps", "int", 8, 1, 32),
            ParamSpec("rotate", "Rotate", "int", 0, 0, 31),
            ParamSpec("division", "Step length", "choice", "1/8", choices=tuple(DIVISIONS)),
            ParamSpec("pulse", "Hit length (of step)", "float", 0.6, 0.05, 1.0, 0.05),
            ParamSpec("shape", "Hit shape", "choice", "decay", choices=HIT_SHAPES),
            ParamSpec("amp_hit", "Hit amp", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("amp_rest", "Rest amp", "float", 0.0, 0.0, 1.0, 0.01),
            ParamSpec("accent", "Off-beat cut", "float", 0.25, 0.0, 1.0, 0.05),
            ParamSpec("freq_hit", "Hit freq", "float", 0.6, 0.0, 1.0, 0.01),
            ParamSpec("freq_rest", "Rest freq", "float", 0.3, 0.0, 1.0, 0.01),
            ParamSpec("b_mode", "Channel B", "choice", "rotate",
                      choices=("same", "rotate", "inverse")),
            ParamSpec("b_rotate", "B rotate (steps)", "int", 1, 0, 31),
        ]

    def render(self, n: int, params: dict, seed: int = 0) -> PulseTrack:
        p = self.merged_params(params)
        bpm = float(params.get("_bpm", 120.0))
        steps = max(1, int(p["steps"]))
        step_s = DIVISIONS.get(p["division"], 0.5) * 60.0 / max(1e-6, bpm)
        t = np.arange(n, dtype=np.float64) / FS
        x = t / step_s
        step_idx = np.floor(x).astype(np.int64)
        pos = x - step_idx
        pulse = float(p["pulse"])
        inside = pos < pulse
        env = np.where(inside, _hit_envelope(p["shape"], np.clip(pos / pulse, 0.0, 1.0)), 0.0)

        base = euclid_pattern(int(p["hits"]), steps, int(p["rotate"]))
        first_hit = int(np.argmax(base)) if base.any() else -1

        def channel(pattern):
            k = step_idx % steps
            hit = pattern[k]
            level = np.where(k == first_hit, 1.0, 1.0 - float(p["accent"]))
            amp = float(p["amp_rest"]) + (float(p["amp_hit"]) - float(p["amp_rest"])) * env * hit * level
            freq = np.where(hit & inside, float(p["freq_hit"]), float(p["freq_rest"]))
            return amp, freq

        mode = p["b_mode"]
        if mode == "inverse":
            pattern_b = ~base
        elif mode == "rotate":
            pattern_b = np.roll(base, int(p["b_rotate"]))
        else:
            pattern_b = base
        l_amp, l_freq = channel(base)
        r_amp, r_freq = channel(pattern_b)
        return PulseTrack(l_amp, r_amp, l_freq, r_freq).clamp()


@register
class RampGenerator(Generator):
    """Slow build-up with optional 'tease' dips at (jittered) intervals."""

    type_id = "ramp"
    label = "Ramp / tease"
    category = "shape"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("amp_start", "Amp start", "float", 0.2, 0.0, 1.0, 0.01),
            ParamSpec("amp_end", "Amp end", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("curve", "Curve (exponent)", "float", 1.5, 0.2, 5.0, 0.1),
            ParamSpec("freq_start", "Freq start", "float", 0.3, 0.0, 1.0, 0.01),
            ParamSpec("freq_end", "Freq end", "float", 0.7, 0.0, 1.0, 0.01),
            ParamSpec("tease_every", "Tease every (s, 0 = off)", "float", 0.0, 0.0, 120.0, 0.5),
            ParamSpec("tease_len", "Tease length (s)", "float", 2.0, 0.1, 30.0, 0.1),
            ParamSpec("tease_depth", "Tease depth", "float", 0.7, 0.0, 1.0, 0.05),
            ParamSpec("jitter", "Tease timing jitter", "float", 0.3, 0.0, 1.0, 0.05),
            ParamSpec("b_lag", "B lag (s)", "float", 0.0, 0.0, 10.0, 0.05),
        ]

    def render(self, n: int, params: dict, seed: int = 0) -> PulseTrack:
        p = self.merged_params(params)
        if n == 0:
            return PulseTrack.zeros(0)
        u = np.linspace(0.0, 1.0, n)
        amp = float(p["amp_start"]) + (float(p["amp_end"]) - float(p["amp_start"])) * \
            u ** float(p["curve"])
        freq = float(p["freq_start"]) + (float(p["freq_end"]) - float(p["freq_start"])) * u

        every, length = float(p["tease_every"]), float(p["tease_len"])
        if every > 0:
            rng = np.random.default_rng(seed)
            dip = np.zeros(n)
            jitter = float(p["jitter"])
            t = every * (1.0 + jitter * rng.uniform(-0.5, 0.5))
            k = max(1, round(length * FS))
            shape = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(k) / max(1, k - 1))
            while t < n / FS:
                i = round(t * FS)
                seg = dip[i:i + k]
                np.maximum(seg, shape[:len(seg)], out=seg)
                t += every * (1.0 + jitter * rng.uniform(-0.5, 0.5))
            amp = amp * (1.0 - float(p["tease_depth"]) * dip)

        lag = round(float(p["b_lag"]) * FS)
        if lag > 0:
            r_amp = np.concatenate([np.full(min(lag, n), amp[0]), amp[:max(0, n - lag)]])
            r_freq = np.concatenate([np.full(min(lag, n), freq[0]), freq[:max(0, n - lag)]])
        else:
            r_amp, r_freq = amp.copy(), freq.copy()
        return PulseTrack(amp, r_amp, freq, r_freq).clamp()

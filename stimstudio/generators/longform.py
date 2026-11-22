"""Long-form and entrainment generators: breathing, heartbeat, eternal accelerando.

* **Breathing pacer** — an inhale / hold / exhale / hold envelope. Slow paced
  breathing near 0.1 Hz (≈ 6 breaths / min, "resonance breathing") maximises
  heart-rate variability (Lehrer & Vaschillo; Clin Neurophysiol 2019). Useful for
  warm-up and cool-down sections.
* **Heartbeat** — "lub-dub" double beats with respiratory variability and a
  tempo ramp; heart-rate feedback alone shifts felt arousal (Valins 1966).
* **Risset accelerando** — the rhythmic Shepard tone: layers of one pulse at
  doubling tempos, cross-faded so the rhythm seems to speed up (or slow down)
  forever while looping seamlessly (Risset; Stowell 2011).
"""

from __future__ import annotations

import math

import numpy as np

from ..core.pulse import FS, PulseTrack
from .base import Generator, ParamSpec, register
from .events import hold_last, place_max
from .noise import smooth_random
from .rhythm import DIVISIONS, HIT_SHAPES, _hit_envelope

# ── breathing ───────────────────────────────────────────────────────────────
#: inhale, hold, exhale, hold — as fractions of one breath
BREATH_PATTERNS = {"resonance (4:6)": (0.4, 0.0, 0.6, 0.0),
                   "box 4-4-4-4": (0.25, 0.25, 0.25, 0.25),
                   "4-7-8": (4 / 19, 7 / 19, 8 / 19, 0.0),
                   "custom": None}


def breath_envelope(u: np.ndarray, parts) -> np.ndarray:
    """0..1 lung-volume-like envelope at cycle positions *u* (0..1)."""
    i, hi, e, _ = parts
    c1, c2, c3 = i, i + hi, i + hi + e
    rise = 0.5 - 0.5 * np.cos(np.pi * np.clip(u / max(i, 1e-9), 0, 1))
    fall = 0.5 + 0.5 * np.cos(np.pi * np.clip((u - c2) / max(e, 1e-9), 0, 1))
    return np.where(u < c1, rise, np.where(u < c2, 1.0, np.where(u < c3, fall, 0.0)))


@register
class BreathGenerator(Generator):
    type_id = "breath"
    label = "Breathing pacer"
    category = "body"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("rate", "Breaths per minute", "float", 6.0, 2.0, 30.0, 0.1),
            ParamSpec("pattern", "Pattern", "choice", "resonance (4:6)",
                      choices=tuple(BREATH_PATTERNS)),
            ParamSpec("inhale", "Custom: inhale", "float", 4.0, 0.1, 20.0, 0.1),
            ParamSpec("hold_in", "Custom: hold after inhale", "float", 0.0, 0.0, 20.0, 0.1),
            ParamSpec("exhale", "Custom: exhale", "float", 6.0, 0.1, 20.0, 0.1),
            ParamSpec("hold_out", "Custom: hold after exhale", "float", 0.0, 0.0, 20.0, 0.1),
            ParamSpec("amp_min", "Amp min", "float", 0.1, 0.0, 1.0, 0.01),
            ParamSpec("amp_max", "Amp max", "float", 0.8, 0.0, 1.0, 0.01),
            ParamSpec("freq_min", "Freq min", "float", 0.2, 0.0, 1.0, 0.01),
            ParamSpec("freq_max", "Freq max", "float", 0.55, 0.0, 1.0, 0.01),
            ParamSpec("freq_mode", "Freq", "choice", "follow",
                      choices=("follow", "inverse", "fixed")),
            ParamSpec("drift", "Tempo drift", "float", 0.0, 0.0, 0.5, 0.01),
            ParamSpec("b_lag", "B lag (s)", "float", 0.0, 0.0, 10.0, 0.05),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        parts = BREATH_PATTERNS.get(p["pattern"])
        if parts is None:
            raw = np.array([float(p[k]) for k in ("inhale", "hold_in", "exhale", "hold_out")])
            parts = tuple(raw / raw.sum())
        rate = float(p["rate"]) / 60.0 * np.ones(n)
        drift = float(p["drift"])
        if drift > 0:
            rate *= 1.0 + drift * 2.0 * (smooth_random(n, 0.05, np.random.default_rng(seed)) - 0.5)
        phase = (np.cumsum(rate) - rate) / FS
        lag = round(float(p["b_lag"]) * FS)

        def channel(ph):
            env = breath_envelope(np.mod(ph, 1.0), parts)
            amp = float(p["amp_min"]) + (float(p["amp_max"]) - float(p["amp_min"])) * env
            mode = p["freq_mode"]
            f = env if mode == "follow" else (1.0 - env if mode == "inverse" else 0.0 * env)
            return amp, float(p["freq_min"]) + (float(p["freq_max"]) - float(p["freq_min"])) * f

        l_amp, l_freq = channel(phase)
        ph_b = phase - lag * float(p["rate"]) / 60.0 / FS if lag else phase
        r_amp, r_freq = channel(ph_b)
        return PulseTrack(l_amp, r_amp, l_freq, r_freq).clamp()


# ── heartbeat ───────────────────────────────────────────────────────────────
@register
class HeartbeatGenerator(Generator):
    type_id = "heartbeat"
    label = "Heartbeat"
    category = "body"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("bpm_start", "Heart rate start (bpm)", "float", 70.0, 30.0, 200.0, 1.0),
            ParamSpec("bpm_end", "Heart rate end (bpm)", "float", 70.0, 30.0, 200.0, 1.0),
            ParamSpec("curve", "Ramp curve", "float", 1.0, 0.2, 5.0, 0.1),
            ParamSpec("hrv", "Breathing variability", "float", 0.05, 0.0, 0.3, 0.01),
            ParamSpec("resp_rate", "Breaths per minute", "float", 12.0, 3.0, 30.0, 0.5),
            ParamSpec("jitter", "Beat jitter", "float", 0.02, 0.0, 0.2, 0.01),
            ParamSpec("lub", "Lub amp", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("dub", "Dub amp", "float", 0.6, 0.0, 1.0, 0.01),
            ParamSpec("beat_ms", "Beat length (ms)", "float", 100.0, 25.0, 300.0, 25.0),
            ParamSpec("amp_rest", "Rest amp", "float", 0.0, 0.0, 1.0, 0.01),
            ParamSpec("freq_beat", "Beat freq", "float", 0.6, 0.0, 1.0, 0.01),
            ParamSpec("freq_rest", "Rest freq", "float", 0.3, 0.0, 1.0, 0.01),
            ParamSpec("b_mode", "Channels", "choice", "same", choices=("same", "lub A / dub B")),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        rng = np.random.default_rng(seed)
        total = max(n / FS, 1e-6)
        b0, b1, curve = float(p["bpm_start"]), float(p["bpm_end"]), float(p["curve"])
        k = max(1, round(float(p["beat_ms"]) / 1000.0 * FS))
        shape = np.exp(-np.arange(k) * (3.0 / k))
        amps = [np.zeros(n), np.zeros(n)]
        active = np.zeros(n, dtype=bool)
        split = p["b_mode"] != "same"
        t = 0.0
        while t * FS < n:
            bpm = b0 + (b1 - b0) * (t / total) ** curve
            rr = 60.0 / bpm * (1.0 + float(p["hrv"]) *
                               math.sin(2 * math.pi * float(p["resp_rate"]) / 60.0 * t)
                               + float(p["jitter"]) * rng.standard_normal())
            rr = max(rr, 0.2)
            s1s2 = 0.32 * math.sqrt(rr)                          # systole shortens with rate
            for when, level, ch in ((t, float(p["lub"]), 0), (t + s1s2, float(p["dub"]), 1)):
                i = round(when * FS)
                for c in ((0, 1) if not split else (ch,)):
                    place_max(amps[c], i, level * shape)
                active[max(0, i):max(0, i + k)] = True
            t += rr
        lo = float(p["amp_rest"])
        l_amp, r_amp = (lo + (1 - lo) * a for a in amps)
        freq = np.where(active, float(p["freq_beat"]), float(p["freq_rest"]))
        return PulseTrack(l_amp, r_amp, freq, freq).clamp()


# ── Risset eternal accelerando ──────────────────────────────────────────────
def risset_onsets(n0: int, layers: int, cycle_s: float, direction: str = "accelerando"):
    """Hits of one cycle: arrays (time, local interval, layer, log-tempo x, weight).

    Layer *k* plays ``n0 · 2^k`` hits per cycle at tempo ``base · 2^(k + τ/C)``;
    weights are a raised cosine over x = (k + τ/C) / layers, so at the cycle end
    layer k has become layer k+1 and the cycle loops seamlessly.
    """
    cols = [[], [], [], [], []]
    for k in range(layers):
        count = n0 * 2 ** k
        j = np.arange(count + 1)
        tau = cycle_s * np.log2(1.0 + j / count)              # beat phase = j
        interval = np.diff(tau)
        if direction == "ritardando":
            times = cycle_s - tau[1:]
            frac = tau[1:] / cycle_s
        else:
            times = tau[:-1]
            frac = tau[:-1] / cycle_s
        x = (k + frac) / layers
        for c, v in zip(cols, (times, interval, np.full(count, k), x,
                               0.5 - 0.5 * np.cos(2 * np.pi * x))):
            c.append(v)
    out = [np.concatenate(c) for c in cols]
    order = np.argsort(out[0], kind="stable")
    return tuple(c[order] for c in out)


@register
class RissetGenerator(Generator):
    type_id = "risset"
    label = "Risset accelerando"
    category = "rhythm"
    tempo_synced = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("direction", "Direction", "choice", "accelerando",
                      choices=("accelerando", "ritardando")),
            ParamSpec("division", "Slowest pulse", "choice", "1/4", choices=tuple(DIVISIONS)),
            ParamSpec("cycle_beats", "Cycle length (slowest hits)", "int", 16, 2, 128),
            ParamSpec("layers", "Layers", "int", 3, 2, 4),
            ParamSpec("pulse", "Hit length (of interval)", "float", 0.45, 0.05, 1.0, 0.05),
            ParamSpec("shape", "Hit shape", "choice", "decay", choices=HIT_SHAPES),
            ParamSpec("amp", "Amplitude", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("freq_slow", "Freq (slow layer)", "float", 0.3, 0.0, 1.0, 0.01),
            ParamSpec("freq_fast", "Freq (fast layer)", "float", 0.8, 0.0, 1.0, 0.01),
            ParamSpec("b_mode", "Channel B", "choice", "same", choices=("same", "opposite")),
        ]

    @staticmethod
    def cycle_seconds(p: dict, bpm: float) -> float:
        base = bpm / 60.0 / DIVISIONS.get(p["division"], 1.0)      # slowest hits per second
        return int(p["cycle_beats"]) * math.log(2.0) / base

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        bpm = float(params.get("_bpm", 120.0))
        cycle = self.cycle_seconds(p, bpm)
        f_lo, f_hi = float(p["freq_slow"]), float(p["freq_fast"])

        def channel(direction):
            times, interval, _, x, w = risset_onsets(int(p["cycle_beats"]), int(p["layers"]),
                                                     cycle, direction)
            amp, num, den = np.zeros(n), np.zeros(n), np.zeros(n)
            for c in range(int(np.ceil(n / FS / cycle)) + 1):
                for t, dt, xx, ww in zip(times + c * cycle, interval, x, w):
                    i = round(t * FS)
                    if i >= n:
                        break
                    k = max(1, round(float(p["pulse"]) * dt * FS))
                    env = _hit_envelope(p["shape"], np.arange(k) / k) * ww
                    place_max(amp, i, env)
                    a, b = max(0, i), min(n, i + k)
                    num[a:b] += env[a - i:b - i] * xx
                    den[a:b] += env[a - i:b - i]
            pos = hold_last(np.where(den > 1e-3, num / np.maximum(den, 1e-9), 0.5), den > 1e-3,
                            0.5)
            k = min(len(pos), round(0.5 * FS))           # glide: freq follows the felt tempo
            if k > 1:
                pos = np.convolve(np.pad(pos, (k, k), mode="edge"), np.ones(k) / k,
                                  mode="same")[k:-k]
            return float(p["amp"]) * amp, f_lo + (f_hi - f_lo) * pos

        direction = p["direction"]
        l_amp, l_freq = channel(direction)
        if p["b_mode"] == "opposite":
            r_amp, r_freq = channel("ritardando" if direction == "accelerando" else "accelerando")
        else:
            r_amp, r_freq = l_amp.copy(), l_freq.copy()
        return PulseTrack(l_amp, r_amp, l_freq, r_freq).clamp()

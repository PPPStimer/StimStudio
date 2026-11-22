"""Noise, chaos and generative-rhythm generators.

* **Colored noise 1/fᵝ** — white (β 0), pink (β 1), brown (β 2) or anything in
  between. Pink noise is closer to biological signals and, applied as sub-
  threshold electrical noise, outperformed white noise (stochastic-resonance
  studies, e.g. Neurosci Lett 2023; Gait Posture 2022).
* **Chaotic attractor** — Lorenz, Rössler or the logistic map: deterministic yet
  never repeating (chaotic signals were compared with noise in the same studies).
* **Groove** — generative 16-step rhythm with a syncopation control. Medium
  syncopation gives the most pleasure and urge to move — an inverted U (Witek
  et al. 2014, PLoS ONE). Syncopation is measured with Longuet-Higgins & Lee's
  metrical weights.
* **Polyrhythm** — m against n beats per bar, one rhythm per channel.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.pulse import FS, PulseTrack
from .base import Generator, ParamSpec, register
from .rhythm import HIT_SHAPES, _hit_envelope


# ── colored noise ───────────────────────────────────────────────────────────
def colored_noise(white: np.ndarray, beta: float, max_hz: float) -> np.ndarray:
    """Shape white noise rows (…, n) to a 1/f^beta spectrum below *max_hz* (40 Hz rate).

    Returns zero-mean, unit-variance rows (zeros for degenerate input).
    """
    n = white.shape[-1]
    if n < 4:
        return np.zeros_like(white, dtype=np.float64)
    spec = np.fft.rfft(white, axis=-1)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    scale = np.zeros_like(f)
    scale[1:] = f[1:] ** (-float(beta) / 2.0)
    scale /= 1.0 + (f / max(float(max_hz), 1e-3)) ** 8          # soft low-pass
    x = np.fft.irfft(spec * scale, n=n, axis=-1)
    x -= x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return np.where(sd > 0, x / np.where(sd > 0, sd, 1.0), 0.0)


@register
class ColoredNoiseGenerator(Generator):
    type_id = "colored_noise"
    label = "Colored noise 1/fᵝ"
    category = "chaos"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("beta", "Color β (0 white · 1 pink · 2 brown)", "float", 1.0, 0.0, 2.5,
                      0.05),
            ParamSpec("max_hz", "Fastest change (Hz)", "float", 4.0, 0.1, 20.0, 0.1),
            ParamSpec("spread", "Spread (σ per range)", "float", 0.33, 0.05, 1.0, 0.01),
            ParamSpec("amp_min", "Amp min", "float", 0.3, 0.0, 1.0, 0.01),
            ParamSpec("amp_max", "Amp max", "float", 0.9, 0.0, 1.0, 0.01),
            ParamSpec("freq_min", "Freq min", "float", 0.2, 0.0, 1.0, 0.01),
            ParamSpec("freq_max", "Freq max", "float", 0.8, 0.0, 1.0, 0.01),
            ParamSpec("correlation", "A/B correlation", "float", 0.5, 0.0, 1.0, 0.01),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((4, n))
        c = float(p["correlation"])
        w[1] = c * w[0] + math.sqrt(max(0.0, 1 - c * c)) * w[1]
        w[3] = c * w[2] + math.sqrt(max(0.0, 1 - c * c)) * w[3]
        x = colored_noise(w, float(p["beta"]), float(p["max_hz"]))
        u = np.clip(0.5 + x * float(p["spread"]) / 2.0, 0.0, 1.0)
        a_lo, a_hi = float(p["amp_min"]), float(p["amp_max"])
        f_lo, f_hi = float(p["freq_min"]), float(p["freq_max"])
        return PulseTrack(a_lo + (a_hi - a_lo) * u[0], a_lo + (a_hi - a_lo) * u[1],
                          f_lo + (f_hi - f_lo) * u[2], f_lo + (f_hi - f_lo) * u[3]).clamp()


# ── chaotic attractors ──────────────────────────────────────────────────────
SYSTEMS = ("lorenz", "rossler", "logistic")


def _deriv(system: str, k: float):
    if system == "rossler":
        a, b, c = 0.2, 0.2, 2.5 + 3.2 * k              # period doubling → chaos at 5.7
        return lambda x, y, z: (-y - z, x + a * y, b + z * (x - c))
    rho = 25.0 + 15.0 * k                              # σ 10, β 8/3
    return lambda x, y, z: (10.0 * (y - x), x * (rho - z) - y, x * y - 8.0 / 3.0 * z)


def attractor_path(system: str, n: int, speed: float, character: float, seed: int) -> np.ndarray:
    """(3, n) trajectory sampled at 40 Hz, each coordinate normalised to 0..1.

    Normalisation uses a fixed calibration run, so the output at a given time does
    not depend on the clip length.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((3, n))
    k = float(np.clip(character, 0.0, 1.0))
    if system == "logistic":
        r = 3.45 + 0.55 * k
        x = 0.1 + 0.8 * rng.uniform()
        for _ in range(300):
            x = r * x * (1 - x)
        rate = 4.0 * max(speed, 1e-3)                   # iterations per second
        count = int(np.ceil(n / FS * rate)) + 4
        it = np.empty(count)
        for i in range(count):
            x = r * x * (1 - x)
            it[i] = x
        it /= r / 4.0                                   # max of the map
        pos = np.arange(n) / FS * rate
        i0 = np.floor(pos).astype(int)
        frac = 0.5 - 0.5 * np.cos(np.pi * (pos - i0))    # cosine interpolation
        for row, lag in enumerate((0, 1, 2)):
            a, b = it[i0 + lag], it[i0 + lag + 1]
            out[row] = a + (b - a) * frac
        return np.clip(out, 0.0, 1.0)

    f = _deriv(system, k)
    units = (3.0 if system == "rossler" else 0.5) * max(speed, 1e-3)   # time units per s
    max_dt = 0.05 if system == "rossler" else 0.01

    def rk4(s, dt):
        x, y, z = s
        k1 = f(x, y, z)
        k2 = f(x + dt / 2 * k1[0], y + dt / 2 * k1[1], z + dt / 2 * k1[2])
        k3 = f(x + dt / 2 * k2[0], y + dt / 2 * k2[1], z + dt / 2 * k2[2])
        k4 = f(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        return (x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                y + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
                z + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))

    s = tuple(float(v) for v in (1.0, 1.0, 1.0) + rng.normal(0.0, 1.0, 3))
    for _ in range(2000):                               # settle onto the attractor
        s = rk4(s, max_dt)
    lo, hi = list(s), list(s)
    for _ in range(8000):                               # calibration: coordinate ranges
        s = rk4(s, max_dt)
        for d in range(3):
            lo[d], hi[d] = min(lo[d], s[d]), max(hi[d], s[d])
    per_pulse = units / FS
    sub = max(1, int(np.ceil(per_pulse / max_dt)))
    dt = per_pulse / sub
    for i in range(n):
        out[:, i] = s
        for _ in range(sub):
            s = rk4(s, dt)
    span = np.array(hi) - np.array(lo)
    out = (out - np.array(lo)[:, None]) / np.where(span > 0, span, 1.0)[:, None]
    return np.clip(out, 0.0, 1.0)


@register
class AttractorGenerator(Generator):
    type_id = "attractor"
    label = "Chaotic attractor"
    category = "chaos"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("system", "System", "choice", "lorenz", choices=SYSTEMS),
            ParamSpec("speed", "Speed", "float", 1.0, 0.05, 10.0, 0.05),
            ParamSpec("character", "Character (calm → wild)", "float", 0.2, 0.0, 1.0, 0.01),
            ParamSpec("amp_min", "Amp min", "float", 0.2, 0.0, 1.0, 0.01),
            ParamSpec("amp_max", "Amp max", "float", 0.9, 0.0, 1.0, 0.01),
            ParamSpec("freq_min", "Freq min", "float", 0.2, 0.0, 1.0, 0.01),
            ParamSpec("freq_max", "Freq max", "float", 0.8, 0.0, 1.0, 0.01),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        x, y, z = attractor_path(p["system"], n, float(p["speed"]), float(p["character"]), seed)
        a_lo, a_hi = float(p["amp_min"]), float(p["amp_max"])
        f_lo, f_hi = float(p["freq_min"]), float(p["freq_max"])
        freq = f_lo + (f_hi - f_lo) * z
        return PulseTrack(a_lo + (a_hi - a_lo) * x, a_lo + (a_hi - a_lo) * y,
                          freq, freq).clamp()


# ── groove (syncopation-controlled rhythm) ──────────────────────────────────
#: Longuet-Higgins & Lee metrical weights of a 4/4 bar in sixteenths.
METRIC_WEIGHTS = np.array([0, -4, -3, -4, -2, -4, -3, -4, -1, -4, -3, -4, -2, -4, -3, -4])


def lhl_syncopation(pattern: np.ndarray) -> float:
    """Longuet-Higgins & Lee syncopation of a cyclic 16-step boolean pattern: every
    note followed by a rest on a stronger position (before the next note) scores
    the weight difference."""
    hits = np.flatnonzero(pattern)
    if len(hits) == 0:
        return 0.0
    total = 0.0
    steps = len(pattern)
    for k, i in enumerate(hits):
        j = hits[(k + 1) % len(hits)] + (steps if k + 1 == len(hits) else 0)
        rests = [(r % steps) for r in range(i + 1, j)]
        if rests:
            strongest = max(METRIC_WEIGHTS[r] for r in rests)
            if strongest > METRIC_WEIGHTS[i]:
                total += strongest - METRIC_WEIGHTS[i]
    return total


def groove_pattern(rng, density: float, syncopation: float) -> np.ndarray:
    """Choose ``density × 16`` onsets, favouring strong beats (syncopation 0) or weak
    ones (syncopation 1)."""
    k = int(np.clip(round(density * 16), 1, 16))
    logits = (1.0 - 2.0 * float(syncopation)) * 1.2 * METRIC_WEIGHTS
    prob = np.exp(logits - logits.max())
    prob /= prob.sum()
    pattern = np.zeros(16, dtype=bool)
    pattern[rng.choice(16, size=k, replace=False, p=prob)] = True
    return pattern


@register
class GrooveGenerator(Generator):
    type_id = "groove"
    label = "Groove (syncopation)"
    category = "rhythm"
    stochastic = True
    tempo_synced = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("syncopation", "Syncopation", "float", 0.45, 0.0, 1.0, 0.01),
            ParamSpec("density", "Density", "float", 0.4, 0.06, 1.0, 0.01),
            ParamSpec("vary_bars", "New variation every (bars)", "int", 2, 1, 64),
            ParamSpec("pulse", "Hit length (of step)", "float", 0.7, 0.05, 1.0, 0.05),
            ParamSpec("shape", "Hit shape", "choice", "decay", choices=HIT_SHAPES),
            ParamSpec("amp_hit", "Hit amp", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("amp_rest", "Rest amp", "float", 0.05, 0.0, 1.0, 0.01),
            ParamSpec("accent", "Metric accent", "float", 0.3, 0.0, 1.0, 0.05),
            ParamSpec("freq_hit", "Hit freq", "float", 0.65, 0.0, 1.0, 0.01),
            ParamSpec("freq_rest", "Rest freq", "float", 0.3, 0.0, 1.0, 0.01),
            ParamSpec("b_mode", "Channel B", "choice", "own",
                      choices=("same", "own", "complement")),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        bpm = float(params.get("_bpm", 120.0))
        step_s = 15.0 / max(bpm, 1e-6)                  # a sixteenth
        x = np.arange(n) / FS / step_s
        step = np.floor(x).astype(np.int64)
        pos = x - step
        bar, k = step // 16, step % 16
        block = bar // max(1, int(p["vary_bars"]))
        dens, sync = float(p["density"]), float(p["syncopation"])
        n_blocks = int(block[-1]) + 1 if n else 0

        def table(which):   # (blocks, 16) patterns, one rng per (seed, block, voice)
            return np.array([groove_pattern(np.random.default_rng([seed, b, which]), dens, sync)
                             for b in range(n_blocks)], dtype=bool).reshape(n_blocks, 16)

        pulse = float(p["pulse"])
        inside = pos < pulse
        env = np.where(inside, _hit_envelope(p["shape"], np.clip(pos / pulse, 0, 1)), 0.0)
        level = 1.0 - float(p["accent"]) * (-METRIC_WEIGHTS[k] / 4.0)

        def channel(which):
            hit = table(1 if which == 1 else 0)[block, k]
            if which == 2:
                hit = ~hit
            lo, hi = float(p["amp_rest"]), float(p["amp_hit"])
            amp = lo + (hi - lo) * env * hit * level
            freq = np.where(hit & inside, float(p["freq_hit"]), float(p["freq_rest"]))
            return amp, freq

        l_amp, l_freq = channel(0)
        which_b = {"same": 0, "own": 1, "complement": 2}.get(p["b_mode"], 1)
        r_amp, r_freq = channel(which_b)
        return PulseTrack(l_amp, r_amp, l_freq, r_freq).clamp()


# ── polyrhythm ──────────────────────────────────────────────────────────────
@register
class PolyrhythmGenerator(Generator):
    type_id = "polyrhythm"
    label = "Polyrhythm"
    category = "rhythm"
    tempo_synced = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("a_beats", "A beats per cycle", "int", 3, 1, 16),
            ParamSpec("b_beats", "B beats per cycle", "int", 2, 1, 16),
            ParamSpec("cycle_beats", "Cycle length (quarter beats)", "int", 4, 1, 32),
            ParamSpec("pulse", "Hit length (of interval)", "float", 0.5, 0.05, 1.0, 0.05),
            ParamSpec("shape", "Hit shape", "choice", "decay", choices=HIT_SHAPES),
            ParamSpec("amp_hit", "Hit amp", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("amp_rest", "Rest amp", "float", 0.0, 0.0, 1.0, 0.01),
            ParamSpec("accent", "Downbeat accent", "float", 0.25, 0.0, 1.0, 0.05),
            ParamSpec("freq_hit", "Hit freq", "float", 0.6, 0.0, 1.0, 0.01),
            ParamSpec("freq_rest", "Rest freq", "float", 0.3, 0.0, 1.0, 0.01),
            ParamSpec("mix", "Channels", "choice", "split", choices=("split", "both")),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        bpm = float(params.get("_bpm", 120.0))
        cycle_s = int(p["cycle_beats"]) * 60.0 / max(bpm, 1e-6)
        u = np.arange(n) / FS / cycle_s
        pulse = float(p["pulse"])

        def voice(m):
            x = u * m
            beat = np.floor(x).astype(np.int64) % m
            pos = x - np.floor(x)
            inside = pos < pulse
            env = np.where(inside, _hit_envelope(p["shape"], np.clip(pos / pulse, 0, 1)), 0.0)
            return env * np.where(beat == 0, 1.0, 1.0 - float(p["accent"])), inside

        ea, ia = voice(int(p["a_beats"]))
        eb, ib = voice(int(p["b_beats"]))
        if p["mix"] == "both":
            ea = eb = np.maximum(ea, eb)
            ia = ib = ia | ib
        lo, hi = float(p["amp_rest"]), float(p["amp_hit"])
        f_hit, f_rest = float(p["freq_hit"]), float(p["freq_rest"])
        return PulseTrack(lo + (hi - lo) * ea, lo + (hi - lo) * eb,
                          np.where(ia, f_hit, f_rest), np.where(ib, f_hit, f_rest)).clamp()

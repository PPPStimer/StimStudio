"""Electrotherapy-inspired generators (TENS / interferential current).

* **Burst train** — burst-mode TENS: trains of high-rate pulses delivered
  1–4 times a second (high internal frequency + low burst rate).
* **Dense–disperse** — alternating low / high pulse frequency ("mixed 2/100 Hz";
  Hamza et al. 1999 — mixed frequencies cut opioid need by 53 % against 32–35 %
  for either alone; alternation delays tolerance, DeSantana et al. 2008).
* **Anti-habituation** — randomly modulated frequency and level with occasional
  novelty events; random modulation needs fewer intensity increases than fixed
  or 6 s / 6 s modulation (Phys Ther 2019), and novel stimuli dishabituate
  (Thompson & Spencer 1966; Rankin et al. 2009).
* **Interferential sweep** — amplitude modulation at a "beat" frequency that
  itself sweeps across a spectrum, as interferential therapy does (limited to
  ~15 Hz by the 40 Hz control rate).
* **Iso-intensity sweep** — frequency sweep with level compensation so the felt
  strength stays even.
"""

from __future__ import annotations

import numpy as np

from ..core.pulse import FS, PulseTrack
from ..core.shapes import shape_wave
from .base import Generator, ParamSpec, register
from .events import iso_compensation, smoothstep


def _cycle_positions(t: np.ndarray, rate: float, jitter: float, rng) -> np.ndarray:
    """Position 0..1 inside the current cycle at times *t*; cycle lengths are
    ``1/rate`` jittered by ±jitter/2. Works for slightly negative times too."""
    period = 1.0 / max(rate, 1e-6)
    t_end = float(t.max()) if len(t) else 0.0
    count = int(np.ceil(t_end / period * 1.6)) + 4
    periods = period * (1.0 + jitter * rng.uniform(-0.5, 0.5, count))
    onsets = np.concatenate([[-2 * period, -period], np.cumsum(np.r_[0.0, periods[:-1]])])
    lengths = np.concatenate([[period, period], periods])
    k = np.clip(np.searchsorted(onsets, t, side="right") - 1, 0, len(onsets) - 1)
    return np.clip((t - onsets[k]) / lengths[k], 0.0, 1.0 - 1e-9)


@register
class BurstGenerator(Generator):
    type_id = "burst"
    label = "Burst train"
    category = "stim"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("rate", "Bursts per second", "float", 2.0, 0.2, 10.0, 0.1),
            ParamSpec("duty", "Burst length (of cycle)", "float", 0.4, 0.05, 0.95, 0.01),
            ParamSpec("attack", "Attack (of burst)", "float", 0.15, 0.0, 1.0, 0.01),
            ParamSpec("release", "Release (of burst)", "float", 0.15, 0.0, 1.0, 0.01),
            ParamSpec("amp_burst", "Burst amp", "float", 0.9, 0.0, 1.0, 0.01),
            ParamSpec("amp_base", "Between bursts amp", "float", 0.0, 0.0, 1.0, 0.01),
            ParamSpec("freq_burst", "Burst freq", "float", 0.9, 0.0, 1.0, 0.01),
            ParamSpec("freq_base", "Between bursts freq", "float", 0.2, 0.0, 1.0, 0.01),
            ParamSpec("b_offset", "B offset (of cycle)", "float", 0.0, 0.0, 1.0, 0.01),
            ParamSpec("jitter", "Timing jitter", "float", 0.0, 0.0, 1.0, 0.05),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        rate = float(p["rate"])
        duty = float(p["duty"])
        t = np.arange(n) / FS
        rng = np.random.default_rng(seed)
        jitter = float(p["jitter"])
        offsets = rng.integers(0, 2 ** 31, 1)   # one shared jitter sequence for A and B
        a_att, a_rel = float(p["attack"]), float(p["release"])

        def channel(shift):
            u = _cycle_positions(t - shift / rate, rate, jitter,
                                 np.random.default_rng(int(offsets[0])))
            inside = u < duty
            v = u / duty
            env = np.ones(n)
            if a_att > 0:
                env = np.minimum(env, v / a_att)
            if a_rel > 0:
                env = np.minimum(env, (1.0 - v) / a_rel)
            env = np.where(inside, np.clip(env, 0.0, 1.0), 0.0)
            lo, hi = float(p["amp_base"]), float(p["amp_burst"])
            amp = lo + (hi - lo) * env
            freq = np.where(inside, float(p["freq_burst"]), float(p["freq_base"]))
            return amp, freq

        l_amp, l_freq = channel(0.0)
        r_amp, r_freq = channel(float(p["b_offset"]))
        return PulseTrack(l_amp, r_amp, l_freq, r_freq).clamp()


def _smooth_square(high: np.ndarray, transition_pulses: int) -> np.ndarray:
    """0/1 signal with raised-cosine transitions of the given length."""
    x = high.astype(np.float64)
    k = int(transition_pulses)
    if k < 2 or len(x) == 0:
        return x
    win = np.hanning(k + 2)[1:-1]
    win /= win.sum()
    pad = np.pad(x, (k, k), mode="edge")
    return np.convolve(pad, win, mode="same")[k:-k]


@register
class DenseDisperseGenerator(Generator):
    type_id = "dense_disperse"
    label = "Dense–disperse"
    category = "stim"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("freq_low", "Low freq", "float", 0.05, 0.0, 1.0, 0.01),
            ParamSpec("freq_high", "High freq", "float", 0.95, 0.0, 1.0, 0.01),
            ParamSpec("low_s", "Low dwell (s)", "float", 3.0, 0.1, 60.0, 0.1),
            ParamSpec("high_s", "High dwell (s)", "float", 3.0, 0.1, 60.0, 0.1),
            ParamSpec("transition_s", "Transition (s)", "float", 0.5, 0.0, 10.0, 0.05),
            ParamSpec("amp", "Amplitude", "float", 0.8, 0.0, 1.0, 0.01),
            ParamSpec("compensate", "Level compensation", "float", 0.3, 0.0, 1.0, 0.01),
            ParamSpec("dip", "Dip at switch", "float", 0.0, 0.0, 1.0, 0.01),
            ParamSpec("b_mode", "Channel B", "choice", "same", choices=("same", "opposite")),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        t = np.arange(n) / FS
        low_s, high_s = float(p["low_s"]), float(p["high_s"])
        high = np.mod(t, low_s + high_s) >= low_s
        h = _smooth_square(high, round(float(p["transition_s"]) * FS))
        lo, hi = float(p["freq_low"]), float(p["freq_high"])

        def channel(level):
            freq = lo + (hi - lo) * level
            amp = float(p["amp"]) * iso_compensation(freq, float(p["compensate"]))
            amp = amp * (1.0 - float(p["dip"]) * 4.0 * level * (1.0 - level))
            return amp, freq

        l_amp, l_freq = channel(h)
        r_amp, r_freq = channel(1.0 - h if p["b_mode"] == "opposite" else h)
        return PulseTrack(l_amp, r_amp, l_freq, r_freq).clamp()


NOVELTY_KINDS = ("surge", "gap", "flutter")


@register
class AntiHabituationGenerator(Generator):
    type_id = "anti_habit"
    label = "Anti-habituation"
    category = "stim"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("amp_min", "Amp min", "float", 0.55, 0.0, 1.0, 0.01),
            ParamSpec("amp_max", "Amp max", "float", 0.9, 0.0, 1.0, 0.01),
            ParamSpec("freq_min", "Freq min", "float", 0.1, 0.0, 1.0, 0.01),
            ParamSpec("freq_max", "Freq max", "float", 0.9, 0.0, 1.0, 0.01),
            ParamSpec("hold_min", "Hold min (s)", "float", 1.0, 0.1, 60.0, 0.1),
            ParamSpec("hold_max", "Hold max (s)", "float", 6.0, 0.1, 60.0, 0.1),
            ParamSpec("glide", "Glide (of hold)", "float", 0.3, 0.0, 1.0, 0.01),
            ParamSpec("unpredictability", "Unpredictability", "float", 0.7, 0.0, 1.0, 0.01),
            ParamSpec("novelty_every", "Novelty every (s, 0 = off)", "float", 20.0, 0.0, 300.0,
                      1.0),
            ParamSpec("novelty_len", "Novelty length (s)", "float", 1.5, 0.2, 10.0, 0.1),
            ParamSpec("link", "A/B link", "float", 0.6, 0.0, 1.0, 0.01),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        rng = np.random.default_rng(seed)
        u = float(p["unpredictability"])
        h_lo, h_hi = sorted((float(p["hold_min"]), float(p["hold_max"])))
        mean_hold = 0.5 * (h_lo + h_hi)

        # segment boundaries shared by both channels
        starts = [0]
        while starts[-1] < n:
            hold = mean_hold + u * (rng.uniform(h_lo, h_hi) - mean_hold)
            starts.append(starts[-1] + max(1, round(hold * FS)))
        m = len(starts) - 1
        # targets: alternate between the range ends when predictable, random when not
        alt = np.arange(m) % 2
        base = rng.uniform(0, 1, (4, m))
        link = float(p["link"])
        base[1] = link * base[0] + (1 - link) * base[1]      # amp B follows amp A
        base[3] = link * base[2] + (1 - link) * base[3]      # freq B follows freq A
        targets = (1 - u) * np.vstack([alt, alt, 1 - alt, 1 - alt]) + u * base

        def curve(row, lo, hi):
            out = np.empty(n)
            prev = targets[row, 0]
            for k in range(m):
                a, b = starts[k], min(starts[k + 1], n)
                if a >= n:
                    break
                g = max(1, round(float(p["glide"]) * (starts[k + 1] - starts[k])))
                s = smoothstep(np.arange(b - a) / g)
                out[a:b] = prev + (targets[row, k] - prev) * s
                prev = targets[row, k]
            return lo + (hi - lo) * out

        amp_lo, amp_hi = float(p["amp_min"]), float(p["amp_max"])
        f_lo, f_hi = float(p["freq_min"]), float(p["freq_max"])
        l_amp, r_amp = curve(0, amp_lo, amp_hi), curve(1, amp_lo, amp_hi)
        l_freq, r_freq = curve(2, f_lo, f_hi), curve(3, f_lo, f_hi)

        every = float(p["novelty_every"])
        if every > 0:
            k = max(2, round(float(p["novelty_len"]) * FS))
            fade = np.hanning(k)
            t = every * rng.uniform(0.5, 1.5)
            while t * FS < n:
                i = round(t * FS)
                kind = NOVELTY_KINDS[rng.integers(len(NOVELTY_KINDS))]
                seg = slice(i, min(n, i + k))
                w = fade[:seg.stop - seg.start]
                if kind == "surge":
                    for arr in (l_amp, r_amp):
                        arr[seg] = arr[seg] + (min(1.0, amp_hi * 1.15) - arr[seg]) * w
                    for arr in (l_freq, r_freq):
                        arr[seg] = arr[seg] + (f_hi - arr[seg]) * w
                elif kind == "gap":
                    for arr in (l_amp, r_amp):
                        arr[seg] = arr[seg] * (1.0 - 0.8 * w)
                else:  # flutter: 5 Hz on/off, A and B in antiphase
                    tt = np.arange(seg.stop - seg.start) / FS
                    sq = (np.mod(tt * 5.0, 1.0) < 0.5).astype(float)
                    l_amp[seg] = l_amp[seg] * (1.0 - w * (1.0 - sq))
                    r_amp[seg] = r_amp[seg] * (1.0 - w * sq)
                t += every * (1.0 + u * rng.uniform(-0.5, 0.5))
        return PulseTrack(l_amp, r_amp, l_freq, r_freq).clamp()


SWEEP_MODES = ("triangle", "ramp_up", "alternate")


@register
class BeatSweepGenerator(Generator):
    type_id = "beat_sweep"
    label = "Interferential sweep"
    category = "stim"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("beat_min", "Beat min (Hz)", "float", 1.0, 0.1, 15.0, 0.1),
            ParamSpec("beat_max", "Beat max (Hz)", "float", 8.0, 0.1, 15.0, 0.1),
            ParamSpec("sweep_s", "Sweep cycle (s)", "float", 12.0, 0.5, 120.0, 0.5),
            ParamSpec("sweep_mode", "Sweep mode", "choice", "triangle", choices=SWEEP_MODES),
            ParamSpec("depth", "Modulation depth", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("amp", "Amplitude", "float", 0.9, 0.0, 1.0, 0.01),
            ParamSpec("freq", "Carrier freq", "float", 0.6, 0.0, 1.0, 0.01),
            ParamSpec("freq_follow", "Freq follows beat", "float", 0.0, 0.0, 1.0, 0.01),
            ParamSpec("b_phase", "B phase (beats)", "float", 0.5, 0.0, 1.0, 0.01),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        t = np.arange(n) / FS
        pos = np.mod(t / float(p["sweep_s"]), 1.0)
        mode = p["sweep_mode"]
        if mode == "ramp_up":
            s = pos
        elif mode == "alternate":        # hold min / hold max with short glides ("6/6")
            s = _smooth_square(pos >= 0.5, round(0.05 * float(p["sweep_s"]) * FS))
        else:
            s = 1.0 - np.abs(2.0 * pos - 1.0)
        b_lo, b_hi = float(p["beat_min"]), float(p["beat_max"])
        beat = b_lo + (b_hi - b_lo) * s
        phase = 2 * np.pi * np.cumsum(beat) / FS
        depth = float(p["depth"])

        def env(extra):
            return float(p["amp"]) * ((1 - depth) + depth * (0.5 + 0.5 * np.cos(phase + extra)))

        freq = float(p["freq"]) + float(p["freq_follow"]) * (1 - float(p["freq"])) * s
        return PulseTrack(env(0.0), env(2 * np.pi * float(p["b_phase"])), freq, freq).clamp()


@register
class IsoSweepGenerator(Generator):
    type_id = "iso_sweep"
    label = "Iso-intensity sweep"
    category = "stim"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("freq_from", "Freq from", "float", 0.1, 0.0, 1.0, 0.01),
            ParamSpec("freq_to", "Freq to", "float", 0.9, 0.0, 1.0, 0.01),
            ParamSpec("sweep_s", "Sweep period (s)", "float", 10.0, 0.2, 300.0, 0.1),
            ParamSpec("shape", "Sweep shape", "choice", "triangle",
                      choices=("triangle", "sine", "saw", "ramp_down")),
            ParamSpec("amp", "Amplitude", "float", 0.85, 0.0, 1.0, 0.01),
            ParamSpec("compensation", "Compensation", "float", 0.5, 0.0, 1.0, 0.01),
            ParamSpec("curve", "Compensation curve", "float", 0.7, 0.2, 3.0, 0.05),
            ParamSpec("b_phase", "B phase (of sweep)", "float", 0.0, 0.0, 1.0, 0.01),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        t = np.arange(n) / FS
        lo, hi = float(p["freq_from"]), float(p["freq_to"])

        def channel(extra):
            freq = lo + (hi - lo) * shape_wave(p["shape"], t / float(p["sweep_s"]) + extra)
            amp = float(p["amp"]) * iso_compensation(freq, float(p["compensation"]),
                                                     float(p["curve"]))
            return amp, freq

        l_amp, l_freq = channel(0.0)
        r_amp, r_freq = channel(float(p["b_phase"]))
        return PulseTrack(l_amp, r_amp, l_freq, r_freq).clamp()

"""Tactile illusions and biomimetic patterns for two channels.

* **Phantom motion** — funneling / phantom sensation: two sites driven together
  are felt as one point between them, placed by their intensity ratio. The
  energy model (A = I·√(1−β), B = I·√β) keeps the felt intensity constant while
  the point moves (Israr & Poupyrev, *Tactile Brush*, CHI 2011; electrotactile
  funneling: *Funnelectro*, 2025). The "steps" mode uses apparent motion with
  Tactile Brush's onset asynchrony SOA = 0.32·duration + 47.3 ms.
* **Cutaneous rabbit** — saltation: a few taps at one site, then at another,
  ~40–80 ms apart, are felt as hops across the skin in between (Geldard &
  Sherrick 1972; ~50 ms maximises the electrocutaneous illusion; gone > 300 ms).
* **Biomimetic touch** — "contacts" coded like natural afferents: strong phasic
  bursts at contact onset and offset (rapidly adapting fibres, ∝ force rate) over
  a weak sustained component (slowly adapting fibres). Biomimetic coding feels
  more natural (Valle et al. 2018, *Neuron*; George et al. 2019, *Sci Robot*).
"""

from __future__ import annotations

import numpy as np

from ..core.pulse import FS, PulseTrack
from .base import Generator, ParamSpec, register
from .events import hold_last, ms_to_pulses, place_max, smoothstep
from .noise import smooth_random

MOTIONS = ("sweep", "ease", "bounce", "random", "steps")


def pan_gains(beta, intensity, law: str = "energy"):
    """Channel levels for a phantom point at *beta* (0 = A, 1 = B)."""
    beta = np.clip(beta, 0.0, 1.0)
    if law == "linear":
        return intensity * (1.0 - beta), intensity * beta
    return intensity * np.sqrt(1.0 - beta), intensity * np.sqrt(beta)


def apparent_motion_soa(duration_s: float) -> float:
    """Tactile Brush onset asynchrony for continuous apparent motion (seconds)."""
    return 0.32 * duration_s + 0.0473


@register
class PhantomGenerator(Generator):
    type_id = "phantom"
    label = "Phantom motion"
    category = "illusion"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("motion", "Motion", "choice", "ease", choices=MOTIONS),
            ParamSpec("speed", "Speed (A↔B per s)", "float", 0.25, 0.01, 10.0, 0.01),
            ParamSpec("intensity", "Intensity", "float", 0.8, 0.0, 1.0, 0.01),
            ParamSpec("width", "Travel width", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("centre", "Centre (A 0 … B 1)", "float", 0.5, 0.0, 1.0, 0.01),
            ParamSpec("law", "Panning law", "choice", "energy", choices=("energy", "linear")),
            ParamSpec("freq", "Freq", "float", 0.5, 0.0, 1.0, 0.01),
            ParamSpec("freq_follow", "Freq follows position", "float", 0.0, -1.0, 1.0, 0.01),
            ParamSpec("points", "Steps: virtual points", "int", 5, 2, 16),
            ParamSpec("step_ms", "Steps: point duration (ms)", "float", 100.0, 25.0, 1000.0,
                      5.0),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        t = np.arange(n) / FS
        speed = float(p["speed"])
        motion = p["motion"]
        width, centre = float(p["width"]), float(p["centre"])
        intensity = float(p["intensity"])

        def place(u):  # 0..1 travel → beta around the centre
            return np.clip(centre + width * (u - 0.5), 0.0, 1.0)

        if motion == "steps":
            pts = int(p["points"])
            order = list(range(pts)) + list(range(pts - 2, 0, -1))
            dur = float(p["step_ms"]) / 1000.0
            soa = apparent_motion_soa(dur)
            e_a, e_b = np.zeros(n), np.zeros(n)
            k = ms_to_pulses(dur * 1000.0)
            env = np.ones(k)
            if k >= 3:
                env[0] = env[-1] = 0.5
            j = 0
            while round(j * soa * FS) < n:
                beta = place(order[j % len(order)] / (pts - 1))
                ga, gb = pan_gains(beta, 1.0, p["law"])
                i = round(j * soa * FS)
                place_max(e_a, i, env * ga ** 2)
                place_max(e_b, i, env * gb ** 2)
                j += 1
            l_amp, r_amp = intensity * np.sqrt(e_a), intensity * np.sqrt(e_b)
            pos_a = np.where(e_a + e_b > 0, e_b / np.maximum(e_a + e_b, 1e-9), np.nan)
            beta = hold_last(np.nan_to_num(pos_a, nan=0.5), ~np.isnan(pos_a), centre)
        else:
            x = t * speed
            if motion == "sweep":
                u = 1.0 - np.abs(2.0 * np.mod(x * 0.5, 1.0) - 1.0)
            elif motion == "bounce":
                f = np.mod(x, 1.0)
                u = 4.0 * f * (1.0 - f)                      # parabolic hop A→B→A
                u = np.where(np.mod(np.floor(x), 2) == 0, u, 1.0 - u)
            elif motion == "random":
                u = smooth_random(n, speed * 2.0, np.random.default_rng(seed)).astype(float)
            else:  # ease
                u = 0.5 - 0.5 * np.cos(np.pi * x)
            beta = place(u)
            l_amp, r_amp = pan_gains(beta, intensity, p["law"])
        freq = float(p["freq"]) + float(p["freq_follow"]) * (beta - 0.5)
        return PulseTrack(l_amp, r_amp, freq, freq).clamp()


DIRECTIONS = ("A → B", "B → A", "ping-pong")


@register
class RabbitGenerator(Generator):
    type_id = "rabbit"
    label = "Cutaneous rabbit"
    category = "illusion"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("taps", "Taps per site", "int", 3, 1, 8),
            ParamSpec("isi_ms", "Tap interval (ms)", "float", 50.0, 25.0, 300.0, 25.0),
            ParamSpec("tap_ms", "Tap length (ms)", "float", 25.0, 25.0, 150.0, 25.0),
            ParamSpec("pause_s", "Pause between runs (s)", "float", 0.6, 0.0, 10.0, 0.05),
            ParamSpec("direction", "Direction", "choice", "ping-pong", choices=DIRECTIONS),
            ParamSpec("mid_site", "Phantom middle site", "bool", True),
            ParamSpec("amp", "Tap amp", "float", 0.9, 0.0, 1.0, 0.01),
            ParamSpec("freq", "Tap freq", "float", 0.7, 0.0, 1.0, 0.01),
            ParamSpec("rest_freq", "Rest freq", "float", 0.4, 0.0, 1.0, 0.01),
        ]

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        isi = ms_to_pulses(float(p["isi_ms"]))
        tap = min(isi, ms_to_pulses(float(p["tap_ms"])))
        pause = round(float(p["pause_s"]) * FS)
        amp = float(p["amp"])
        mid = amp * np.sqrt(0.5)
        sites = [(amp, 0.0)] + ([(mid, mid)] if p["mid_site"] else []) + [(0.0, amp)]
        l_amp, r_amp = np.zeros(n), np.zeros(n)
        active = np.zeros(n, dtype=bool)
        i, run = 0, 0
        while i < n:
            forward = p["direction"] == "A → B" or (p["direction"] == "ping-pong" and run % 2 == 0)
            for ga, gb in (sites if forward else sites[::-1]):
                for _ in range(int(p["taps"])):
                    l_amp[i:i + tap] = ga
                    r_amp[i:i + tap] = gb
                    active[i:i + isi] = True        # whole run keeps the tap freq
                    i += isi
            i += pause
            run += 1
        freq = np.where(active, float(p["freq"]), float(p["rest_freq"]))
        return PulseTrack(l_amp, r_amp, freq, freq).clamp()


TOUCH_CHANNELS = ("both", "alternate", "random")


@register
class TouchGenerator(Generator):
    type_id = "touch"
    label = "Biomimetic touch"
    category = "illusion"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("rate", "Touches per minute", "float", 20.0, 1.0, 120.0, 1.0),
            ParamSpec("hold_s", "Hold (s)", "float", 1.5, 0.1, 10.0, 0.05),
            ParamSpec("jitter", "Timing jitter", "float", 0.4, 0.0, 1.0, 0.05),
            ParamSpec("press_s", "Press time (s)", "float", 0.15, 0.025, 2.0, 0.025),
            ParamSpec("release_s", "Release time (s)", "float", 0.2, 0.025, 2.0, 0.025),
            ParamSpec("force", "Force", "float", 0.8, 0.0, 1.0, 0.01),
            ParamSpec("sustain", "Sustained (SA) level", "float", 0.35, 0.0, 1.0, 0.01),
            ParamSpec("onset_gain", "Onset/offset (RA) gain", "float", 1.0, 0.0, 1.0, 0.01),
            ParamSpec("freq_sa", "Sustained freq", "float", 0.3, 0.0, 1.0, 0.01),
            ParamSpec("freq_ra", "Transient freq", "float", 0.9, 0.0, 1.0, 0.01),
            ParamSpec("channel", "Channels", "choice", "both", choices=TOUCH_CHANNELS),
        ]

    def render(self, n, params, seed=0):
        if n == 0:
            return PulseTrack.zeros(0)
        p = self.merged_params(params)
        rng = np.random.default_rng(seed)
        force = float(p["force"])
        press = max(1, round(float(p["press_s"]) * FS))
        release = max(1, round(float(p["release_s"]) * FS))
        jitter = float(p["jitter"])
        forces = [np.zeros(n), np.zeros(n)]
        interval = 60.0 / float(p["rate"])
        t, k = interval * rng.uniform(0.1, 0.6), 0
        while t * FS < n:
            hold = max(1, round(float(p["hold_s"]) * (1 + jitter * rng.uniform(-0.5, 0.5)) * FS))
            shape = np.concatenate([smoothstep(np.arange(1, press + 1) / press),
                                    np.ones(hold),
                                    1.0 - smoothstep(np.arange(1, release + 1) / release)])
            mode = p["channel"]
            chans = (0, 1) if mode == "both" else \
                ((k % 2,) if mode == "alternate" else (int(rng.integers(2)),))
            for c in chans:
                place_max(forces[c], round(t * FS), force * shape)
            gap = interval * (1 + jitter * rng.uniform(-0.5, 0.5))
            t += max(gap, (len(shape) + 1) / FS)
            k += 1

        peak_rate = 1.5 * force / (min(press, release) / FS) if force > 0 else 1.0
        tail = np.exp(-np.arange(round(0.12 * FS)) / (0.04 * FS))   # RA burst decay

        def channel(f):
            slope = np.abs(np.gradient(f)) if len(f) > 1 else np.zeros_like(f)
            ra = np.clip(slope * FS / peak_rate, 0.0, 1.0)
            ra = np.clip(np.convolve(ra, tail)[:n], 0.0, 1.0) * float(p["onset_gain"])
            sa = float(p["sustain"]) * f
            amp = np.clip(sa + ra * max(force, 0.3), 0.0, 1.0)
            mix = np.where(amp > 0, ra / np.maximum(ra + sa, 1e-9), 0.0)
            freq = float(p["freq_sa"]) + (float(p["freq_ra"]) - float(p["freq_sa"])) * mix
            return amp, freq

        l_amp, l_freq = channel(forces[0])
        r_amp, r_freq = channel(forces[1])
        return PulseTrack(l_amp, r_amp, l_freq, r_freq).clamp()

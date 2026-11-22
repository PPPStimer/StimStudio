"""Triphase generators: position trajectories for triphase tracks.

They produce triphase arrays (columns volume, x, y, freq — x / y as 0..1, see
``core.triphase``) and are only offered for triphase tracks (``domain``).
"""

from __future__ import annotations

import numpy as np

from ..core.pulse import FS, PulseTrack
from ..core.triphase import ELECTRODE_ANGLES_DEG, xy_to_unit
from .base import Generator, ParamSpec, register
from .noise import smooth_random

SWING_SHAPES = ("sine", "triangle", "square")


def _track(n, vol, x, y, freq) -> PulseTrack:
    """Build the triphase PulseTrack; positions are clamped into the unit disk."""
    x = np.broadcast_to(np.asarray(x, dtype=np.float64), (n,)).copy()
    y = np.broadcast_to(np.asarray(y, dtype=np.float64), (n,)).copy()
    r = np.hypot(x, y)
    over = r > 1.0
    x[over] /= r[over]
    y[over] /= r[over]
    def full(v):  # writable copy (broadcast views are read-only)
        return np.array(np.broadcast_to(v, (n,)), dtype=np.float32)
    return PulseTrack(full(vol), xy_to_unit(x), xy_to_unit(y), full(freq)).clamp()


class TriphaseGenerator(Generator):
    domain = "triphase"
    category = "triphase"

    @staticmethod
    def common_specs():
        return [ParamSpec("volume", "Volume", "float", 0.8, 0.0, 1.0, 0.01),
                ParamSpec("freq", "Pulse freq", "float", 0.5, 0.0, 1.0, 0.01)]


@register
class TriPoint(TriphaseGenerator):
    type_id = "tri_point"
    label = "Fixed position"

    @classmethod
    def param_specs(cls):
        return [ParamSpec("x", "Position X", "float", 0.0, -1.0, 1.0, 0.01),
                ParamSpec("y", "Position Y", "float", 0.0, -1.0, 1.0, 0.01)] + cls.common_specs()

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        return _track(n, float(p["volume"]), float(p["x"]), float(p["y"]), float(p["freq"]))


@register
class TriCircle(TriphaseGenerator):
    type_id = "tri_circle"
    label = "Circle"

    @classmethod
    def param_specs(cls):
        return [ParamSpec("radius", "Radius", "float", 0.8, 0.0, 1.0, 0.01),
                ParamSpec("speed", "Speed (rev/s, − = reverse)", "float", 0.25, -5.0, 5.0, 0.01),
                ParamSpec("start", "Start angle (°)", "float", 90.0, 0.0, 360.0, 1.0),
                ParamSpec("wobble", "Radius wobble", "float", 0.0, 0.0, 1.0, 0.01),
                ParamSpec("wobble_speed", "Wobble speed (Hz)", "float", 0.5, 0.0, 10.0, 0.05),
                ] + cls.common_specs()

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        t = np.arange(n) / FS
        ang = np.radians(float(p["start"])) + 2 * np.pi * float(p["speed"]) * t
        r = float(p["radius"]) * (1.0 - float(p["wobble"]) *
                                  (0.5 - 0.5 * np.cos(2 * np.pi * float(p["wobble_speed"]) * t)))
        return _track(n, float(p["volume"]), r * np.cos(ang), r * np.sin(ang), float(p["freq"]))


@register
class TriPendulum(TriphaseGenerator):
    type_id = "tri_pendulum"
    label = "Pendulum"

    @classmethod
    def param_specs(cls):
        return [ParamSpec("angle_a", "From angle (°)", "float", 150.0, 0.0, 360.0, 1.0),
                ParamSpec("angle_b", "To angle (°)", "float", 30.0, 0.0, 360.0, 1.0),
                ParamSpec("radius", "Radius", "float", 0.9, 0.0, 1.0, 0.01),
                ParamSpec("through_centre", "Swing through the centre", "bool", False),
                ParamSpec("speed", "Swings per second", "float", 0.5, 0.01, 10.0, 0.01),
                ParamSpec("shape", "Swing shape", "choice", "sine", choices=SWING_SHAPES),
                ] + cls.common_specs()

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        ph = np.arange(n) / FS * float(p["speed"])
        frac = np.mod(ph, 1.0)
        if p["shape"] == "triangle":
            u = 1.0 - np.abs(2.0 * frac - 1.0)
        elif p["shape"] == "square":
            u = (frac >= 0.5).astype(float)
        else:
            u = 0.5 - 0.5 * np.cos(2 * np.pi * frac)
        r = float(p["radius"])
        a = np.radians(float(p["angle_a"]))
        b = np.radians(float(p["angle_b"]))
        if p["through_centre"]:          # straight line between the two points
            x = r * ((1 - u) * np.cos(a) + u * np.cos(b))
            y = r * ((1 - u) * np.sin(a) + u * np.sin(b))
        else:                            # along the arc (shortest way round)
            d = (b - a + np.pi) % (2 * np.pi) - np.pi
            ang = a + u * d
            x, y = r * np.cos(ang), r * np.sin(ang)
        return _track(n, float(p["volume"]), x, y, float(p["freq"]))


@register
class TriSpiral(TriphaseGenerator):
    type_id = "tri_spiral"
    label = "Spiral"

    @classmethod
    def param_specs(cls):
        return [ParamSpec("r_min", "Inner radius", "float", 0.1, 0.0, 1.0, 0.01),
                ParamSpec("r_max", "Outer radius", "float", 0.9, 0.0, 1.0, 0.01),
                ParamSpec("spin", "Spin (rev/s)", "float", 0.5, -5.0, 5.0, 0.01),
                ParamSpec("breathe", "In/out (Hz)", "float", 0.1, 0.0, 5.0, 0.01),
                ] + cls.common_specs()

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        t = np.arange(n) / FS
        lo, hi = float(p["r_min"]), float(p["r_max"])
        r = lo + (hi - lo) * (0.5 - 0.5 * np.cos(2 * np.pi * float(p["breathe"]) * t))
        ang = np.pi / 2 + 2 * np.pi * float(p["spin"]) * t
        return _track(n, float(p["volume"]), r * np.cos(ang), r * np.sin(ang), float(p["freq"]))


@register
class TriWander(TriphaseGenerator):
    type_id = "tri_wander"
    label = "Wander (random)"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [ParamSpec("speed", "Speed (pts/s)", "float", 0.5, 0.05, 10.0, 0.05),
                ParamSpec("radius", "Max radius", "float", 0.9, 0.0, 1.0, 0.01),
                ParamSpec("vol_depth", "Volume variation", "float", 0.2, 0.0, 1.0, 0.01),
                ] + cls.common_specs()

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        rng = np.random.default_rng(seed)
        speed, rad = float(p["speed"]), float(p["radius"])
        x = smooth_random(n, speed, rng, -rad, rad).astype(np.float64)
        y = smooth_random(n, speed, rng, -rad, rad).astype(np.float64)
        v = float(p["volume"]) * (1.0 - float(p["vol_depth"]) * smooth_random(n, speed, rng))
        return _track(n, v, x, y, float(p["freq"]))


ELECTRODE_SEQ = {"L → R → N": "LRN", "L → R": "LR", "N → L → N → R": "NLNR",
                 "around (L R N)": "LRN", "centre ↔ L": "CL", "centre ↔ R": "CR",
                 "centre ↔ N": "CN"}


@register
class TriElectrodes(TriphaseGenerator):
    type_id = "tri_electrodes"
    label = "Electrode hop"

    @classmethod
    def param_specs(cls):
        return [ParamSpec("sequence", "Sequence", "choice", "L → R → N",
                          choices=tuple(ELECTRODE_SEQ)),
                ParamSpec("step", "Step (s)", "float", 1.0, 0.05, 30.0, 0.05),
                ParamSpec("glide", "Glide (of step)", "float", 0.3, 0.0, 1.0, 0.01),
                ParamSpec("radius", "Radius", "float", 1.0, 0.0, 1.0, 0.01),
                ] + cls.common_specs()

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        r = float(p["radius"])
        pts = []
        for c in ELECTRODE_SEQ.get(p["sequence"], "LRN"):
            if c == "C":
                pts.append((0.0, 0.0))
            else:
                a = np.radians(ELECTRODE_ANGLES_DEG[c])
                pts.append((r * np.cos(a), r * np.sin(a)))
        pts = np.array(pts)
        pos = np.arange(n) / FS / float(p["step"])
        k = np.floor(pos).astype(int)
        frac = pos - k
        glide = float(p["glide"])
        # hold for (1 - glide) of the step, then glide to the next point
        u = np.clip((frac - (1 - glide)) / max(glide, 1e-9), 0, 1) if glide > 0 else 0 * frac
        u = u * u * (3 - 2 * u)
        a, b = pts[k % len(pts)], pts[(k + 1) % len(pts)]
        xy = a + (b - a) * u[:, None]
        return _track(n, float(p["volume"]), xy[:, 0], xy[:, 1], float(p["freq"]))


@register
class TriLissajous(TriphaseGenerator):
    """Lissajous figure or rose curve: dense, slowly evolving coverage of the triangle."""

    type_id = "tri_lissajous"
    label = "Lissajous / rose"

    @classmethod
    def param_specs(cls):
        return [ParamSpec("mode", "Curve", "choice", "lissajous", choices=("lissajous", "rose")),
                ParamSpec("a", "Lissajous: X ratio", "int", 3, 1, 12),
                ParamSpec("b", "Lissajous: Y ratio", "int", 2, 1, 12),
                ParamSpec("delta", "Lissajous: phase (°)", "float", 90.0, 0.0, 360.0, 1.0),
                ParamSpec("petals", "Rose: k", "int", 3, 1, 12),
                ParamSpec("speed", "Speed (cycles/s)", "float", 0.1, 0.005, 5.0, 0.005),
                ParamSpec("radius", "Radius", "float", 0.9, 0.0, 1.0, 0.01),
                ] + cls.common_specs()

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        th = 2 * np.pi * float(p["speed"]) * np.arange(n) / FS
        r = float(p["radius"])
        if p["mode"] == "rose":
            rr = r * np.cos(int(p["petals"]) * th)
            x, y = rr * np.cos(th + np.pi / 2), rr * np.sin(th + np.pi / 2)
        else:
            x = r * np.sin(int(p["a"]) * th + np.radians(float(p["delta"])))
            y = r * np.sin(int(p["b"]) * th)
        return _track(n, float(p["volume"]), x, y, float(p["freq"]))


@register
class TriRabbit(TriphaseGenerator):
    """Cutaneous-rabbit saltation between electrodes: short taps hop from one
    electrode to the next (Geldard & Sherrick 1972; ~50 ms intervals)."""

    type_id = "tri_rabbit"
    label = "Saltation (rabbit)"

    @classmethod
    def param_specs(cls):
        return [ParamSpec("sequence", "Sequence", "choice", "L → R → N",
                          choices=tuple(ELECTRODE_SEQ)),
                ParamSpec("taps", "Taps per electrode", "int", 3, 1, 8),
                ParamSpec("isi_ms", "Tap interval (ms)", "float", 50.0, 25.0, 300.0, 25.0),
                ParamSpec("tap_ms", "Tap length (ms)", "float", 25.0, 25.0, 150.0, 25.0),
                ParamSpec("pause_s", "Pause between runs (s)", "float", 0.5, 0.0, 10.0, 0.05),
                ParamSpec("radius", "Radius", "float", 1.0, 0.0, 1.0, 0.01),
                ParamSpec("rest_volume", "Volume between taps", "float", 0.0, 0.0, 1.0, 0.01),
                ] + cls.common_specs()

    def render(self, n, params, seed=0):
        p = self.merged_params(params)
        r = float(p["radius"])
        pts = [(0.0, 0.0) if c == "C" else
               (r * np.cos(np.radians(ELECTRODE_ANGLES_DEG[c])),
                r * np.sin(np.radians(ELECTRODE_ANGLES_DEG[c])))
               for c in ELECTRODE_SEQ.get(p["sequence"], "LRN")]
        isi = max(1, round(float(p["isi_ms"]) / 1000.0 * FS))
        tap = min(isi, max(1, round(float(p["tap_ms"]) / 1000.0 * FS)))
        pause = round(float(p["pause_s"]) * FS)
        vol = np.full(n, float(p["rest_volume"]) * float(p["volume"]))
        x, y = np.zeros(n), np.zeros(n)
        i = 0
        while i < n:
            for px, py in pts:
                for _ in range(int(p["taps"])):
                    vol[i:i + tap] = float(p["volume"])
                    x[i:i + isi], y[i:i + isi] = px, py
                    i += isi
            x[i:i + pause], y[i:i + pause] = pts[-1]
            i += pause
        return _track(n, vol, x, y, float(p["freq"]))


@register
class TriAttractor(TriphaseGenerator):
    """A chaotic attractor (Lorenz / Rössler) projected onto the position disc; its
    third coordinate shapes the volume."""

    type_id = "tri_attractor"
    label = "Chaotic attractor"
    stochastic = True

    @classmethod
    def param_specs(cls):
        return [ParamSpec("system", "System", "choice", "lorenz", choices=("lorenz", "rossler")),
                ParamSpec("speed", "Speed", "float", 0.5, 0.05, 10.0, 0.05),
                ParamSpec("character", "Character (calm → wild)", "float", 0.2, 0.0, 1.0, 0.01),
                ParamSpec("radius", "Radius", "float", 0.9, 0.0, 1.0, 0.01),
                ParamSpec("vol_depth", "Volume from 3rd axis", "float", 0.3, 0.0, 1.0, 0.01),
                ] + cls.common_specs()

    def render(self, n, params, seed=0):
        from .complexity import attractor_path
        p = self.merged_params(params)
        a, b, c = attractor_path(p["system"], n, float(p["speed"]), float(p["character"]), seed)
        r = float(p["radius"])
        vol = float(p["volume"]) * (1.0 - float(p["vol_depth"]) * (1.0 - c))
        return _track(n, vol, r * (2 * a - 1), r * (2 * b - 1), float(p["freq"]))

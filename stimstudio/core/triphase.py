"""Triphase (three-electrode) model.

Electrodes L and R are driven by the two output channels, N is common. With
the output voltages as phasors ``a`` (L) and ``b`` (R), N at 0 V and equal
resistance between every electrode pair, the current through each electrode is

    I_L = 2a − b,    I_R = 2b − a,    I_N = −(a + b)        (they sum to zero)

and the sensation at an electrode follows ``|I|``. A position ``(x, y)`` in the
unit disk picks the three current magnitudes; because the currents sum to
zero they form a closed triangle, which fixes the phasors (law of cosines):

* centre → equal currents, 120° apart: balanced three-phase;
* towards an electrode → more current through it;
* towards the middle of an edge → less current through the opposite electrode.

Electrodes sit on the unit circle: N at −90° (bottom), L at 150°, R at 30°.

**Audio** plays the phasors on a carrier (true phase steering). **Pulse
outputs** (Howl / Coyote: channel A to L, B to R, both returns tied to N) have
no phase, so they get the magnitudes ``|a|`` and ``|b|``.

Triphase data on a track is a pulse-shaped (N, 4) array with columns
``volume, x, y, freq`` — x / y stored as 0..1 (``0.5`` = centre) so the usual
lanes, envelopes and fades apply unchanged.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass

import numpy as np

ELECTRODES = ("N", "L", "R")
ELECTRODE_ANGLES_DEG = {"N": -90.0, "L": 150.0, "R": 30.0}
_UNIT = {k: (math.cos(math.radians(a)), math.sin(math.radians(a)))
         for k, a in ELECTRODE_ANGLES_DEG.items()}

#: Column meaning of triphase arrays.
VOL, X, Y, FREQ = 0, 1, 2, 3
TRI_LANE_NAMES = ("Volume", "Position X", "Position Y", "Pulse freq")


@dataclass(frozen=True)
class Calibration:
    """Per-setup corrections, stored in the project.

    *center_x / center_y* move the neutral point; *gain_n / gain_l / gain_r*
    compensate electrode sensitivity; *reach* (0.25..1) is how far the edge of
    the disk pushes current towards one electrode (0.5 keeps every position
    reachable; above that the extremes are clipped to what is physical).
    """

    center_x: float = 0.0
    center_y: float = 0.0
    gain_n: float = 1.0
    gain_l: float = 1.0
    gain_r: float = 1.0
    reach: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Calibration":
        d = d or {}
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})


DEFAULT_CAL = Calibration()


def unit_to_xy(u) -> np.ndarray:
    """Stored 0..1 → position −1..1."""
    return 2.0 * np.asarray(u, dtype=np.float64) - 1.0


def xy_to_unit(v) -> np.ndarray:
    return np.clip((np.asarray(v, dtype=np.float64) + 1.0) / 2.0, 0.0, 1.0)


def currents(x, y, cal: Calibration = DEFAULT_CAL):
    """Electrode current magnitudes ``(m_N, m_L, m_R)`` for positions (x, y)."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    shape = x.shape
    x = x.ravel() - cal.center_x
    y = y.ravel() - cal.center_y
    r = np.hypot(x, y)
    scale = np.where(r > 1.0, 1.0 / np.maximum(r, 1e-12), 1.0)     # stay in the disk
    x, y = x * scale, y * scale
    gains = {"N": cal.gain_n, "L": cal.gain_l, "R": cal.gain_r}
    mn, ml, mr = (np.maximum((1.0 + cal.reach * (x * _UNIT[k][0] + y * _UNIT[k][1])) * gains[k],
                             0.0) for k in ELECTRODES)
    # physical limit: no current may exceed the other two together
    total = mn + ml + mr
    for arr, (o1, o2) in ((mn, (ml, mr)), (ml, (mn, mr)), (mr, (mn, ml))):
        over = arr > total / 2.0
        arr[over] = (o1 + o2)[over]
    return mn.reshape(shape), ml.reshape(shape), mr.reshape(shape)


def phasors(mn, ml, mr):
    """Output phasors ``(a, b)`` whose electrode currents have the magnitudes given.

    The currents close a triangle: I_L = m_L, I_R = m_R·e^{iγ}, I_N = −(I_L + I_R),
    with cos γ from |I_N| = m_N; then a = (2·I_L + I_R) / 3, b = (2·I_R + I_L) / 3.
    """
    ml = np.asarray(ml, dtype=np.float64)
    mr = np.asarray(mr, dtype=np.float64)
    mn = np.asarray(mn, dtype=np.float64)
    denom = 2.0 * ml * mr
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_g = np.where(denom > 1e-12, (mn ** 2 - ml ** 2 - mr ** 2) / denom, -0.5)
    gamma = np.arccos(np.clip(cos_g, -1.0, 1.0))
    i_l = ml.astype(np.complex128)
    i_r = mr * np.exp(1j * gamma)
    return (2.0 * i_l + i_r) / 3.0, (2.0 * i_r + i_l) / 3.0


@functools.lru_cache(maxsize=32)
def _norm(cal: Calibration) -> float:
    """Largest output amplitude over the whole disk (keeps signals ≤ 1)."""
    g = np.linspace(-1.0, 1.0, 81)
    xx, yy = np.meshgrid(g, g)
    ang = np.linspace(0.0, 2 * np.pi, 1441)     # the extremes lie on the rim
    xs = np.concatenate([xx.ravel(), np.cos(ang)])
    ys = np.concatenate([yy.ravel(), np.sin(ang)])
    a, b = phasors(*currents(xs, ys, cal))
    return float(max(np.abs(a).max(), np.abs(b).max(), 1e-9))


def output_phasors(vol, x, y, cal: Calibration = DEFAULT_CAL):
    """Normalised output phasors for volume 0..1 and position (x, y): |a|, |b| ≤ 1."""
    a, b = phasors(*currents(x, y, cal))
    k = np.asarray(vol, dtype=np.float64) / _norm(cal)
    a, b = a * k, b * k
    # hard limit (the sampled normalisation can miss the true peak by a hair)
    peak = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
    return a / peak, b / peak


def phasors_to_tri(a, b, cal: Calibration = DEFAULT_CAL):
    """Inverse of :func:`output_phasors`: phasors ``(a, b)`` → ``(vol, x, y)``.

    The electrode currents ``c_k = |I_k| / gain_k`` are ``s·(1 + reach·(p · u_k))``
    with ``s = vol / norm``; the unit vectors ``u_k`` sum to zero and
    ``Σ u_k u_kᵀ = 3/2·I``, so ``s`` is their mean and the position their
    projection. Exact wherever the forward model was not clipped (outside the
    disk or at the physical limit); silence maps to the centre.
    """
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    gains = {"N": cal.gain_n, "L": cal.gain_l, "R": cal.gain_r}
    c = {"N": np.abs(a + b) / gains["N"], "L": np.abs(2 * a - b) / gains["L"],
         "R": np.abs(2 * b - a) / gains["R"]}
    s = (c["N"] + c["L"] + c["R"]) / 3.0
    k = np.where(s > 1e-9, 2.0 / (3.0 * np.maximum(s, 1e-9) * cal.reach), 0.0)
    x = k * sum(c[e] * _UNIT[e][0] for e in ELECTRODES)
    y = k * sum(c[e] * _UNIT[e][1] for e in ELECTRODES)
    r = np.hypot(x, y)
    scale = np.where(r > 1.0, 1.0 / np.maximum(r, 1e-12), 1.0)     # back into the disk
    return s * _norm(cal), x * scale + cal.center_x, y * scale + cal.center_y


def to_ab(vol, x, y, cal: Calibration = DEFAULT_CAL):
    """Pulse-output amplitudes (channel A → L, B → R, returns tied to N)."""
    a, b = output_phasors(vol, x, y, cal)
    return np.abs(a), np.abs(b)


def tri_to_pulses(tri: np.ndarray, cal: Calibration = DEFAULT_CAL) -> np.ndarray:
    """A triphase array (volume, x, y, freq; 0..1) → ordinary (N, 4) A/B pulses."""
    out = np.zeros((len(tri), 4), dtype=np.float32)
    if len(tri):
        amp_a, amp_b = to_ab(tri[:, VOL], unit_to_xy(tri[:, X]), unit_to_xy(tri[:, Y]), cal)
        out[:, 0], out[:, 1] = amp_a, amp_b
        out[:, 2] = out[:, 3] = tri[:, FREQ]
    return np.clip(out, 0.0, 1.0)


def electrode_levels(vol, x, y, cal: Calibration = DEFAULT_CAL):
    """Relative sensation per electrode (|I|) for display, normalised to the
    largest current over the disk."""
    a, b = output_phasors(vol, x, y, cal)
    return {"N": np.abs(a + b), "L": np.abs(2 * a - b), "R": np.abs(2 * b - a)}


def mix_tri(layers) -> np.ndarray | None:
    """Combine several triphase arrays: volumes add, position and frequency are
    volume-weighted averages. *layers* is ``[(array, gain), ...]`` or
    ``[(array, gain, freq_weight, freq_on), ...]``: *freq_weight* (n,) is the
    volume that weights the layer's frequency (default: its volume), and with
    *freq_on* False the layer's frequency is left out (a frequency nobody
    supplies holds its last value, see ``core.mixer``)."""
    from .mixer import hold_frequency
    layers = [(*ly, None, True)[:4] for ly in layers if ly[0] is not None]
    if not layers:
        return None
    n = len(layers[0][0])
    vol = np.zeros(n)
    acc = np.zeros((n, 2))
    w_sum = np.zeros(n)
    f_num = np.zeros(n)
    f_den = np.zeros(n)
    for t, g, fw, f_on in layers:
        v = t[:, VOL].astype(np.float64) * g
        w = v + 1e-6
        vol += v
        acc += w[:, None] * t[:, X:Y + 1]
        w_sum += w
        if f_on:
            wf = w if fw is None else np.asarray(fw, dtype=np.float64) * g + 1e-6
            f_num += wf * t[:, FREQ]
            f_den += wf
    out = np.zeros((n, 4), dtype=np.float32)
    out[:, VOL] = np.clip(vol, 0.0, 1.0)
    out[:, X:Y + 1] = acc / w_sum[:, None]
    out[:, FREQ] = hold_frequency(f_num, f_den)
    return out

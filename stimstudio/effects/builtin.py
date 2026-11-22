"""Built-in effects.

The first twelve are ported from HWL Viewer unchanged; ``tests/data/golden.npz``
pins their output, so any numeric change there is a deliberate behaviour
change. The rest are new for StimStudio.
"""

from __future__ import annotations

import numpy as np

from ..core.params import ParamSpec as P
from ..core.pulse import AMP_COLS, FREQ_COLS, L_AMP, L_FREQ, R_AMP, R_FREQ
from ..core.shapes import SHAPES, shape_wave
from .registry import EffectContext, register_effect

def num(p: dict, key: str):
    """A parameter as float, or as an (n,) array when it is automated."""
    v = p[key]
    return v if isinstance(v, np.ndarray) else float(v)


def col(v):
    """Broadcast an automated (n,) parameter against an (n, k) column block."""
    return v[:, None] if isinstance(v, np.ndarray) else v


#: Effects whose numeric behaviour is pinned by the HWL Viewer golden file,
#: in the HWL Viewer order.
LEGACY_EFFECTS = (
    "channel_swap", "invert_frequencies", "phase_shift", "frequency_adjust",
    "scale_amplitude", "scale_frequency", "amplitude_feel", "frequency_feel",
    "random_amplitude_noise", "random_frequency_noise",
    "smooth_amplitude", "smooth_frequency",
)


# ── ported from HWL Viewer ──────────────────────────────────────────────────
@register_effect("channel_swap", "Swap channels", category="channels")
def channel_swap(r: np.ndarray, p: dict, ctx: EffectContext) -> None:
    # A-only: copy B→A; B-only: copy A→B; both: full swap
    src = r.copy()
    if ctx.a:
        r[:, L_AMP] = src[:, R_AMP]
        r[:, L_FREQ] = src[:, R_FREQ]
    if ctx.b:
        r[:, R_AMP] = src[:, L_AMP]
        r[:, R_FREQ] = src[:, L_FREQ]


@register_effect("invert_frequencies", "Invert frequencies", category="frequency")
def invert_frequencies(r, p, ctx) -> None:
    r[:, ctx.freqs] = 1.0 - r[:, ctx.freqs]


@register_effect("phase_shift", "Phase shift", [
    P("us", "Shift (µs)", "int", 0, 0, 9_975_000, 25_000),
], category="time")
def phase_shift(r, p, ctx) -> None:
    shift = int(round(float(p["us"]) * ctx.fs / 1_000_000.0))
    if shift:
        cols = ctx.amps + ctx.freqs
        r[:, cols] = np.roll(r[:, cols], shift, axis=0)


@register_effect("frequency_adjust", "Frequency adjust", [
    P("amount", "Amount", "float", 0.0, -1.0, 1.0, 0.01),
], category="frequency", automatable=("amount",))
def frequency_adjust(r, p, ctx) -> None:
    r[:, ctx.freqs] = np.clip(r[:, ctx.freqs] + col(num(p, "amount")), 0.0, 1.0)


@register_effect("scale_amplitude", "Scale amplitude", [
    P("value", "Value", "float", 1.0, 0.0, 4.0, 0.01),
], automatable=("value",))
def scale_amplitude(r, p, ctx) -> None:
    r[:, ctx.amps] = np.minimum(r[:, ctx.amps] * col(num(p, "value")), 1.0)


@register_effect("scale_frequency", "Scale frequency", [
    P("value", "Value", "float", 1.0, 0.0, 2.0, 0.01),
], category="frequency", automatable=("value",))
def scale_frequency(r, p, ctx) -> None:
    r[:, ctx.freqs] = np.minimum(r[:, ctx.freqs] * col(num(p, "value")), 1.0)


@register_effect("amplitude_feel", "Amplitude feel", [
    P("feel", "Feel (exponent)", "float", 1.0, 0.1, 50.0, 0.05),
], automatable=("feel",))
def amplitude_feel(r, p, ctx) -> None:
    r[:, ctx.amps] = r[:, ctx.amps] ** col(num(p, "feel"))


@register_effect("frequency_feel", "Frequency feel", [
    P("feel", "Feel (exponent)", "float", 1.0, 0.1, 50.0, 0.05),
], category="frequency", automatable=("feel",))
def frequency_feel(r, p, ctx) -> None:
    r[:, ctx.freqs] = r[:, ctx.freqs] ** col(num(p, "feel"))


def _noise_pair(n: int, amount: float, speed: float, fs: int, seed: int):
    """Two smoothly interpolated random curves in [-amount, amount].

    The RNG is re-seeded per effect and the left curve is drawn first, so the
    noise is reproducible for a given parameter set.
    """
    rng = np.random.default_rng(seed)
    if speed > 0 and n > 1:
        num_kf = max(2, int(n * speed / fs) + 2)
        xs = np.linspace(0, n - 1, num_kf)
        grid = np.arange(n)
        left = np.interp(grid, xs, rng.uniform(-amount, amount, num_kf))
        right = np.interp(grid, xs, rng.uniform(-amount, amount, num_kf))
        return left, right
    zeros = np.zeros(n)
    return zeros, zeros


def _scaled_noise(ctx, p):
    """The noise pair; an automated amount scales unit-amount noise per pulse
    (the scalar path is kept exactly as ported, for the golden reference)."""
    amount = num(p, "amount")
    if isinstance(amount, np.ndarray):
        nl, nr = _noise_pair(ctx.n, 1.0, float(p["speed"]), ctx.fs, int(p["seed"]))
        return nl * amount, nr * amount
    return _noise_pair(ctx.n, amount, float(p["speed"]), ctx.fs, int(p["seed"]))


@register_effect("random_amplitude_noise", "Amp noise", [
    P("amount", "Amount", "float", 0.2, 0.0, 1.0, 0.01),
    P("speed", "Speed (pts/s)", "float", 4.0, 0.0, 20.0, 0.01),
    P("skip_zeros", "Skip zeros", "bool", False),
    P("threshold", "Zero threshold", "float", 0.001, 0.0, 0.5, 0.001),
    P("seed", "Seed", "int", 42, 0, 2**31 - 1, 1),
], automatable=("amount",))
def random_amplitude_noise(r, p, ctx) -> None:
    noise_l, noise_r = _scaled_noise(ctx, p)
    skip_zeros = bool(p.get("skip_zeros", False))
    threshold = float(p.get("threshold", 0.001))
    for col, noise, on in ((L_AMP, noise_l, ctx.a), (R_AMP, noise_r, ctx.b)):
        if not on:
            continue
        cur = r[:, col]
        new = np.clip(cur + noise, 0.0, 1.0)
        r[:, col] = np.where(cur < threshold, cur, new) if skip_zeros else new


@register_effect("random_frequency_noise", "Freq noise", [
    P("amount", "Amount", "float", 0.2, 0.0, 1.0, 0.01),
    P("speed", "Speed (pts/s)", "float", 4.0, 0.0, 20.0, 0.01),
    P("seed", "Seed", "int", 42, 0, 2**31 - 1, 1),
], category="frequency", automatable=("amount",))
def random_frequency_noise(r, p, ctx) -> None:
    noise_l, noise_r = _scaled_noise(ctx, p)
    for col, noise, on in ((L_FREQ, noise_l, ctx.a), (R_FREQ, noise_r, ctx.b)):
        if on:
            r[:, col] = np.clip(r[:, col] + noise, 0.0, 1.0)


def _smooth(r, cols, window_s, ctx) -> None:
    k = max(1, int(float(window_s) * ctx.fs))
    kernel = np.ones(k) / k
    for col in cols:
        # "same" yields max(n, k) samples, so trim when the window is longer
        # than the segment itself.
        r[:, col] = np.convolve(r[:, col], kernel, "same")[:ctx.n]


@register_effect("smooth_amplitude", "Smooth amplitude", [
    P("window", "Window (s)", "float", 0.1, 0.01, 2.0, 0.01),
])
def smooth_amplitude(r, p, ctx) -> None:
    _smooth(r, ctx.amps, p["window"], ctx)


@register_effect("smooth_frequency", "Smooth frequency", [
    P("window", "Window (s)", "float", 0.1, 0.01, 2.0, 0.01),
], category="frequency")
def smooth_frequency(r, p, ctx) -> None:
    _smooth(r, ctx.freqs, p["window"], ctx)


# ── new effects ─────────────────────────────────────────────────────────────
def _lfo(ctx: EffectContext, p: dict, side: int) -> np.ndarray:
    """Unipolar LFO in [0, 1] on the timeline clock; B can be phase-offset."""
    phase = float(p["phase"]) + (float(p["b_phase"]) if side == 1 else 0.0)
    cyc = ctx.times() * float(p["rate"]) + phase
    return shape_wave(p["shape"], cyc, float(p.get("duty", 0.5))).astype(np.float64)


_LFO_PARAMS = (
    P("shape", "Shape", "choice", "sine", choices=SHAPES),
    P("rate", "Rate (Hz)", "float", 1.0, 0.01, 20.0, 0.01),
    P("phase", "Phase", "float", 0.0, 0.0, 1.0, 0.01),
    P("b_phase", "B phase offset", "float", 0.0, 0.0, 1.0, 0.01),
    P("duty", "Pulse duty", "float", 0.5, 0.01, 0.99, 0.01),
)


@register_effect("tremolo", "Tremolo (amp LFO)", (
    P("depth", "Depth", "float", 0.5, 0.0, 1.0, 0.01),
) + _LFO_PARAMS, automatable=("depth",))
def tremolo(r, p, ctx) -> None:
    depth = num(p, "depth")
    for side, col in enumerate(AMP_COLS):
        if col in ctx.amps:
            r[:, col] *= 1.0 - depth * (1.0 - _lfo(ctx, p, side))


@register_effect("vibrato", "Vibrato (freq LFO)", (
    P("depth", "Depth", "float", 0.1, 0.0, 1.0, 0.01),
) + _LFO_PARAMS, category="frequency", automatable=("depth",))
def vibrato(r, p, ctx) -> None:
    depth = num(p, "depth")
    for side, col in enumerate(FREQ_COLS):
        if col in ctx.freqs:
            r[:, col] = np.clip(r[:, col] + depth * (2.0 * _lfo(ctx, p, side) - 1.0), 0.0, 1.0)


GATE_DIVISIONS = ("1/1", "1/2", "1/4", "1/8", "1/16", "1/4T", "1/8T")


def _division_beats(div: str) -> float:
    triplet = div.endswith("T")
    denom = int(div.rstrip("T").split("/")[1])
    beats = 4.0 / denom
    return beats * 2.0 / 3.0 if triplet else beats


@register_effect("gate", "Tempo gate", [
    P("division", "Division", "choice", "1/8", choices=GATE_DIVISIONS),
    P("duty", "Open fraction", "float", 0.5, 0.05, 1.0, 0.01),
    P("floor", "Closed level", "float", 0.0, 0.0, 1.0, 0.01),
    P("edge", "Edge (s)", "float", 0.02, 0.0, 0.5, 0.005),
    P("b_offset", "B offset (steps)", "float", 0.0, 0.0, 1.0, 0.25),
], category="time", automatable=("floor",))
def gate(r, p, ctx) -> None:
    """Tempo-synced chopper. Edges are softened with a short linear ramp."""
    step_s = _division_beats(p["division"]) * 60.0 / max(1e-3, float(ctx.bpm))
    duty = float(p["duty"])
    floor = num(p, "floor")
    edge = max(1e-6, float(p["edge"]))
    for side, col in enumerate(AMP_COLS):
        if col not in ctx.amps:
            continue
        offset = float(p["b_offset"]) if side == 1 else 0.0
        pos = np.mod(ctx.times() / step_s - offset, 1.0) * step_s   # seconds into the step
        open_s = duty * step_s
        rise = np.clip(pos / edge, 0.0, 1.0)
        fall = np.clip((open_s - pos) / edge, 0.0, 1.0)
        g = np.minimum(rise, fall)
        r[:, col] *= floor + (1.0 - floor) * g


@register_effect("freq_range", "Frequency range", [
    P("low", "Low", "float", 0.0, 0.0, 1.0, 0.01),
    P("high", "High", "float", 1.0, 0.0, 1.0, 0.01),
], category="frequency", automatable=("low", "high"))
def freq_range(r, p, ctx) -> None:
    """Remap frequency 0..1 into [low, high]."""
    lo, hi = col(num(p, "low")), col(num(p, "high"))
    r[:, ctx.freqs] = lo + (hi - lo) * np.clip(r[:, ctx.freqs], 0.0, 1.0)


@register_effect("slew_limit", "Slew limiter", [
    P("rise", "Max rise (/s)", "float", 2.0, 0.05, 40.0, 0.05),
    P("fall", "Max fall (/s, 0 = free)", "float", 0.0, 0.0, 40.0, 0.05),
], category="safety", automatable=("rise",))
def slew_limit(r, p, ctx) -> None:
    """Limit how fast amplitude may change. A safety stage for the master bus:
    a sudden 0 → 1 step becomes a ramp of ``1 / rise`` seconds."""
    up = np.broadcast_to(np.asarray(num(p, "rise"), dtype=np.float64) / ctx.fs, (ctx.n,))
    fall = float(p["fall"])
    down = fall / ctx.fs if fall > 0 else np.inf
    for col in ctx.amps:
        x = r[:, col]
        out = np.empty_like(x)
        prev = 0.0  # the array starts from silence
        for i in range(ctx.n):
            v = x[i]
            if v > prev + up[i]:
                v = prev + up[i]
            elif v < prev - down:
                v = prev - down
            out[i] = v
            prev = v
        r[:, col] = out


@register_effect("triphase_pan", "Triphase pan", [
    P("angle", "Angle (°)", "float", 90.0, 0.0, 360.0, 1.0),
    P("radius", "Radius", "float", 0.0, 0.0, 1.0, 0.01),
    P("reach", "Reach", "float", 0.5, 0.25, 1.0, 0.05),
], category="channels", automatable=("angle", "radius"))
def triphase_pan(r, p, ctx) -> None:
    """Place a normal track on a three-electrode setup (A → L, B → R, returns to N).

    The louder of A/B becomes the level; the position (angle / radius on the
    electrode circle: N at 270°, L at 150°, R at 30°) redistributes it between
    the channels with the triphase model. Frequencies are kept.
    """
    from ..core.triphase import Calibration, to_ab
    level = np.maximum(r[:, L_AMP], r[:, R_AMP])
    ang = np.radians(num(p, "angle"))
    rad = num(p, "radius")
    amp_a, amp_b = to_ab(level, rad * np.cos(ang), rad * np.sin(ang),
                         Calibration(reach=float(p["reach"])))
    r[:, L_AMP], r[:, R_AMP] = amp_a, amp_b

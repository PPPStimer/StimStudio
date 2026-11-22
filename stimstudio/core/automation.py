"""Evaluate lane automation envelopes and apply them to a track."""

from __future__ import annotations

import numpy as np

from .model import Envelope, Lane
from .pulse import FS, LANE_KEYS


def evaluate(env: Envelope, times: np.ndarray) -> np.ndarray:
    """Envelope value at each time. Before the first / after the last point the
    end values are held."""
    if not env.points:
        return np.ones_like(times, dtype=np.float64)
    pts = sorted(env.points)
    ts = np.array([p[0] for p in pts], dtype=np.float64)
    vs = np.clip(np.array([p[1] for p in pts], dtype=np.float64), 0.0, 1.0)
    if len(pts) == 1 or env.interp == "linear":
        return np.interp(times, ts, vs)
    # index of the segment start for every time
    i = np.clip(np.searchsorted(ts, times, side="right") - 1, 0, len(ts) - 1)
    if env.interp == "hold":
        return vs[i]
    # smooth: smoothstep between neighbouring points
    j = np.minimum(i + 1, len(ts) - 1)
    span = np.where(ts[j] > ts[i], ts[j] - ts[i], 1.0)
    u = np.clip((times - ts[i]) / span, 0.0, 1.0)
    u = u * u * (3.0 - 2.0 * u)
    out = vs[i] + (vs[j] - vs[i]) * u
    out[times <= ts[0]] = vs[0]
    out[times >= ts[-1]] = vs[-1]
    return out


def apply_lanes(pulses: np.ndarray, lanes: dict[str, Lane], t0: float = 0.0,
                fs: int = FS, mute: bool = True) -> np.ndarray:
    """Apply lane mutes and envelopes in place; returns *pulses*.

    A muted amplitude lane is zeroed (a muted frequency lane has nothing to
    zero: it only leaves the master mix, see ``core.mixer``). With *mute*
    False the mutes are left to the caller."""
    times = None
    for col, key in enumerate(LANE_KEYS):
        lane = lanes.get(key)
        if lane is None:
            continue
        if mute and lane.muted and key.startswith("amp"):
            pulses[:, col] = 0.0
            continue
        env = lane.envelope
        if env is None or not env.enabled or not env.points:
            continue
        if times is None:
            times = t0 + np.arange(len(pulses), dtype=np.float64) / fs
        v = evaluate(env, times)
        x = pulses[:, col].astype(np.float64)
        if env.mode == "multiply":
            x = x * v
        elif env.mode == "offset":
            x = x + (2.0 * v - 1.0)
        elif env.mode == "replace":
            x = v
        else:
            raise ValueError(f"unknown envelope mode: {env.mode!r}")
        pulses[:, col] = np.clip(x, 0.0, 1.0)
    return pulses

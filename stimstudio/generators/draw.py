"""Draw generator: user-drawn amplitude and frequency envelopes per channel."""
from __future__ import annotations

import numpy as np

from ..core.pulse import PulseTrack
from .base import Generator, register


def _decode(s: str, n: int, default: float = 0.5) -> np.ndarray:
    try:
        pts = [float(x) for x in str(s).split(",") if x.strip()]
    except ValueError:
        pts = []
    if not pts:
        return np.full(n, default, dtype=np.float32)
    src = np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0)
    if len(src) == n:
        return src
    xp = np.linspace(0.0, 1.0, len(src))
    xi = np.linspace(0.0, 1.0, n)
    return np.interp(xi, xp, src).astype(np.float32)


@register
class DrawGenerator(Generator):
    type_id = "draw"
    label = "Draw"
    category = "manual"

    @classmethod
    def param_specs(cls):
        return []

    def render(self, n: int, params: dict, seed: int = 0) -> PulseTrack:
        return PulseTrack(
            _decode(params.get("amp_a", ""), n, 0.5),
            _decode(params.get("amp_b", ""), n, 0.5),
            _decode(params.get("freq_a", ""), n, 0.5),
            _decode(params.get("freq_b", ""), n, 0.5),
        )

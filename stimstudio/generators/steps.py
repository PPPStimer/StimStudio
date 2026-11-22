"""Step-sequencer generator: N equal steps, each with its own amp/freq value."""
from __future__ import annotations

import numpy as np

from ..core.pulse import PulseTrack
from .base import Generator, ParamSpec, register


def decode_step_vals(raw: str, steps: int, default: float = 0.5) -> np.ndarray:
    try:
        pts = [float(x) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        pts = []
    if not pts:
        return np.full(steps, default, dtype=np.float32)
    arr = np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0)
    if len(arr) >= steps:
        return arr[:steps]
    return np.pad(arr, (0, steps - len(arr)), mode="edge").astype(np.float32)


def step_vals_to_samples(vals: np.ndarray, n: int, transition: str) -> np.ndarray:
    steps = len(vals)
    if steps == 0:
        return np.full(n, 0.5, dtype=np.float32)
    if transition == "smooth":
        centers = np.linspace(0.5 / steps, 1.0 - 0.5 / steps, steps)
        xi = np.linspace(0.0, 1.0, n)
        return np.interp(xi, centers, vals).astype(np.float32)
    out = np.empty(n, dtype=np.float32)
    for i, v in enumerate(vals):
        s = int(round(i * n / steps))
        e = int(round((i + 1) * n / steps))
        out[s:e] = v
    return out


@register
class StepGenerator(Generator):
    type_id = "steps"
    label = "Step Sequencer"
    category = "manual"

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("steps",      "Steps",         "int",    8,        1, 32),
            ParamSpec("transition", "Transition",    "choice", "smooth",
                      choices=("hard", "smooth")),
            ParamSpec("phase_b",    "B step offset", "int",    0,        0, 31),
        ]

    def render(self, n: int, params: dict, seed: int = 0) -> PulseTrack:
        p = self.merged_params(params)
        steps = max(1, int(p["steps"]))
        transition = str(p["transition"])
        phase_b = int(p["phase_b"]) % steps

        amp_a  = decode_step_vals(params.get("amp_a_vals",  ""), steps, 0.5)
        amp_b  = decode_step_vals(params.get("amp_b_vals",  ""), steps, 0.5)
        freq_a = decode_step_vals(params.get("freq_a_vals", ""), steps, 0.5)
        freq_b = decode_step_vals(params.get("freq_b_vals", ""), steps, 0.5)

        if phase_b:
            amp_b  = np.roll(amp_b,  phase_b)
            freq_b = np.roll(freq_b, phase_b)

        return PulseTrack(
            step_vals_to_samples(amp_a,  n, transition),
            step_vals_to_samples(amp_b,  n, transition),
            step_vals_to_samples(freq_a, n, transition),
            step_vals_to_samples(freq_b, n, transition),
        )

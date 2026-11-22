"""Regenerate the bundled clip presets in stimstudio/resources/presets.

    python tools/make_presets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from stimstudio.core.model import Clip, GeneratorSource  # noqa: E402
from stimstudio.core.render import segment_stages  # noqa: E402
from stimstudio.io.presets import BUNDLED_DIR, save_preset  # noqa: E402


def csv(values) -> str:
    return ",".join(f"{v:.4f}" for v in values)


def presets() -> dict[str, Clip]:
    out: dict[str, Clip] = {}

    def add(folder, name, type_id, length, params=None, seed=0, **kw):
        clip = Clip(GeneratorSource(type_id, params or {}, seed), length_s=length, name=name, **kw)
        out[f"{folder}/{name}"] = clip

    add("Shapes", "Warm-up wave", "waveform", 30.0,
        {"amp_speed": 0.2, "amp_min": 0.25, "amp_max": 0.8, "freq_speed": 0.07,
         "freq_min": 0.2, "freq_max": 0.5}, fade_in_s=3.0, fade_out_s=2.0)
    add("Shapes", "Ping-pong", "channels", 20.0,
        {"amp_speed": 0.5, "pan_speed": 0.5, "mode": "ping_pong", "freq": 0.45})
    add("Shapes", "Slow build with teases", "ramp", 90.0,
        {"amp_start": 0.15, "amp_end": 0.95, "curve": 1.8, "tease_every": 15.0,
         "tease_len": 3.0, "tease_depth": 0.8}, seed=1, fade_out_s=2.0)
    add("Shapes", "Wandering noise", "noise", 30.0,
        {"kind": "fractal", "speed": 0.6, "amp_min": 0.35}, seed=4)
    add("Rhythm", "Euclid 3-8 gallop", "euclid", 16.0,
        {"hits": 3, "steps": 8, "division": "1/8", "shape": "decay"})
    add("Rhythm", "Euclid 5-8 drive", "euclid", 16.0,
        {"hits": 5, "steps": 8, "division": "1/16", "shape": "square", "pulse": 0.5,
         "amp_rest": 0.15})
    add("Rhythm", "Euclid 7-16 ping-pong", "euclid", 16.0,
        {"hits": 7, "steps": 16, "division": "1/16", "b_mode": "inverse", "pulse": 0.8,
         "shape": "sine"})
    add("Rhythm", "Groove medium syncopation", "groove", 16.0, {"syncopation": 0.45}, seed=3)
    add("Rhythm", "Polyrhythm 3-2", "polyrhythm", 16.0, {"a_beats": 3, "b_beats": 2})
    add("Rhythm", "Risset eternal accelerando", "risset", 60.0, {"cycle_beats": 16},
        fade_in_s=2.0, fade_out_s=2.0)
    add("Electrotherapy", "Burst train 2 Hz", "burst", 30.0, {"rate": 2.0, "duty": 0.4})
    add("Electrotherapy", "Dense-disperse 3-3", "dense_disperse", 60.0,
        {"low_s": 3.0, "high_s": 3.0, "compensate": 0.3})
    add("Electrotherapy", "Anti-habituation long", "anti_habit", 180.0,
        {"unpredictability": 0.8, "novelty_every": 25.0}, seed=2, fade_in_s=3.0, fade_out_s=3.0)
    add("Electrotherapy", "Interferential sweep 1-10 Hz", "beat_sweep", 30.0,
        {"beat_min": 1.0, "beat_max": 10.0, "sweep_s": 15.0})
    add("Illusions", "Phantom glide", "phantom", 30.0, {"motion": "ease", "speed": 0.2})
    add("Illusions", "Apparent motion steps", "phantom", 20.0,
        {"motion": "steps", "points": 5, "step_ms": 100.0})
    add("Illusions", "Cutaneous rabbit", "rabbit", 20.0, {"taps": 3, "isi_ms": 50.0})
    add("Illusions", "Biomimetic touches", "touch", 40.0, {"rate": 15.0}, seed=1)
    add("Noise & chaos", "Pink noise", "colored_noise", 60.0, {"beta": 1.0}, seed=5)
    add("Noise & chaos", "Lorenz drift", "attractor", 60.0,
        {"system": "lorenz", "speed": 0.6}, seed=2)
    add("Breath & body", "Resonance breathing 6 per min", "breath", 120.0,
        {"rate": 6.0}, fade_in_s=5.0, fade_out_s=5.0)
    add("Breath & body", "Heartbeat 60 to 110", "heartbeat", 90.0,
        {"bpm_start": 60.0, "bpm_end": 110.0, "curve": 1.5}, seed=1)
    add("Manual", "Step pattern 8", "steps", 8.0,
        {"steps": 8, "transition": "hard",
         "amp_a_vals": csv([1, .3, .7, .3, 1, .3, .9, 0]),
         "amp_b_vals": csv([.3, 1, .3, .7, .3, 1, 0, .9]),
         "freq_a_vals": csv([.5] * 8), "freq_b_vals": csv([.6] * 8)})
    swell = 0.5 - 0.45 * np.cos(np.linspace(0, 2 * np.pi, 64))
    add("Manual", "Drawn swell", "draw", 12.0,
        {"amp_a": csv(swell), "amp_b": csv(swell[::-1]),
         "freq_a": csv(np.linspace(0.3, 0.7, 64)), "freq_b": csv(np.linspace(0.7, 0.3, 64))})
    add("Howl", "Licks", "licks", 60.0, seed=7, fade_in_s=1.0, fade_out_s=2.0)
    add("Howl", "Relentless", "relentless", 60.0, seed=3, fade_in_s=2.0, fade_out_s=2.0)
    stages = segment_stages("milkmaster", {}, 5, 240.0)
    if len(stages) > 3:
        start, end = stages[2]
        add("Howl", "Milkmaster stage 3", "milkmaster", end - start, seed=5, trim_s=start,
            fade_in_s=0.5, fade_out_s=0.5)
    return out


def main() -> int:
    for rel, clip in presets().items():
        path = save_preset(clip, BUNDLED_DIR / rel)
        print("wrote", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""A small demo project that exercises generators, effects, automation and mixing."""

from __future__ import annotations

from .core.model import Clip, Envelope, GeneratorSource, Lane, Project, Track
from .effects import get_effect


def build_demo_project() -> Project:
    base = Track(name="Base", color="#4ec9b0", clips=[
        Clip(GeneratorSource("waveform", {"amp_speed": 0.25, "amp_min": 0.3, "freq_speed": 0.1}),
             start_s=0.0, length_s=20.0, fade_in_s=2.0, fade_out_s=1.0, name="Warm-up"),
        Clip(GeneratorSource("licks", {}, seed=7),
             start_s=19.0, length_s=31.0, fade_in_s=1.0, fade_out_s=3.0, name="Licks"),
    ])
    base.lanes["amp_r"] = Lane(envelope=Envelope(points=[(0.0, 0.6), (25.0, 1.0), (50.0, 0.8)]))

    pulse = Track(name="Pulse", color="#dcdcaa", gain=0.5, clips=[
        Clip(GeneratorSource("noise", {"kind": "smooth", "speed": 2.0}, seed=3),
             start_s=10.0, length_s=30.0, fade_in_s=4.0, fade_out_s=4.0, name="Noise bed"),
    ], effects=[get_effect("gate").make(division="1/8", duty=0.6)])

    return Project(
        name="Demo", bpm=100.0, tracks=[base, pulse],
        master_effects=[get_effect("slew_limit").make(rise=4.0)],
    )

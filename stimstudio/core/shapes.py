"""Unipolar periodic waveform shapes shared by generators and effects."""

from __future__ import annotations

import numpy as np

SHAPES = ("sine", "triangle", "saw", "ramp_down", "square", "pulse")


def shape_wave(shape: str, phase: np.ndarray, duty: float = 0.5) -> np.ndarray:
    """Unipolar [0, 1] waveform sampled at fractional cycle positions *phase*."""
    p = np.mod(phase, 1.0)
    if shape == "sine":
        return (0.5 - 0.5 * np.cos(2.0 * np.pi * p)).astype(np.float32)
    if shape == "triangle":
        return (1.0 - np.abs(2.0 * p - 1.0)).astype(np.float32)
    if shape == "saw":
        return p.astype(np.float32)
    if shape == "ramp_down":
        return (1.0 - p).astype(np.float32)
    if shape == "square":
        return (p < 0.5).astype(np.float32)
    if shape == "pulse":
        return (p < float(duty)).astype(np.float32)
    raise ValueError(f"unknown shape: {shape!r}")

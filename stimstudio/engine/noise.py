"""NoiseGenerator: OpenSimplex-based noise on a circular path (ported)."""

from __future__ import annotations

import math

import opensimplex

from . import mathutils as mu


class NoiseGenerator:
    def __init__(self, seed=None):
        if seed is None:
            seed = mu.rng().getrandbits(62)
        self._noise = opensimplex.OpenSimplex(int(seed))

    def get_noise(self, time, rotation, radius, axis, shift_result):
        circle_x = radius * math.cos(rotation)
        circle_y = radius * math.sin(rotation)
        if axis == 0:
            x0, y0, z0 = time, circle_x, circle_y
            x1, y1, z1 = time, -circle_x, -circle_y
        elif axis == 1:
            x0, y0, z0 = circle_x, time, circle_y
            x1, y1, z1 = -circle_x, time, -circle_y
        elif axis == 2:
            x0, y0, z0 = circle_x, circle_y, time
            x1, y1, z1 = -circle_x, -circle_y, time
        else:
            raise ValueError("Axis must be 0, 1, or 2")
        noise_a = self._noise.noise3(x0, y0, z0)
        noise_b = self._noise.noise3(x1, y1, z1)
        if shift_result:
            return (noise_a + 1) / 2, (noise_b + 1) / 2
        return noise_a, noise_b

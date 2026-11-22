"""WaveShape / WavePoint / CyclicalWave / JitterHandler.

Ported from Howl's Utils.kt (WaveShape) and CyclicalWave.kt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import mathutils as mu

SMALL_AMOUNT = 0.00001
TMAX = 1.0 - SMALL_AMOUNT

HERMITE = "HERMITE"
LINEAR = "LINEAR"


@dataclass
class WavePoint:
    time: float
    position: float
    slope: float | None = None


class WaveShape:
    def __init__(self, name, points, interpolation_type=HERMITE):
        for p in points:
            if not (0.0 <= p.time < 1.0):
                raise ValueError("All times must be in [0.0, 1.0)")
        sorted_points = sorted(points, key=lambda p: p.time)
        # distinctBy time
        seen = {}
        for p in sorted_points:
            if p.time not in seen:
                seen[p.time] = p
        sorted_points = list(seen.values())
        if len(sorted_points) < 2:
            raise ValueError("Shape must contain at least two unique points")

        self.name = name
        self.interpolation_type = interpolation_type
        if interpolation_type == HERMITE:
            if all(p.slope is not None for p in sorted_points):
                self.points = sorted_points
            else:
                slopes = self._compute_monotone_slopes(sorted_points)
                self.points = [
                    WavePoint(p.time, p.position, p.slope if p.slope is not None else slopes[i])
                    for i, p in enumerate(sorted_points)
                ]
        else:
            self.points = sorted_points

    @staticmethod
    def _compute_monotone_slopes(points):
        n = len(points)
        slopes = [0.0] * n
        d = [0.0] * n
        for i in range(n):
            nxt = 0 if i == n - 1 else i + 1
            if i == n - 1:
                h = (1.0 + points[nxt].time) - points[i].time
            else:
                h = points[nxt].time - points[i].time
            d[i] = (points[nxt].position - points[i].position) / h

        for i in range(n):
            prev = n - 1 if i == 0 else i - 1
            if d[prev] * d[i] <= 0.0:
                slopes[i] = 0.0
            else:
                slopes[i] = (d[prev] + d[i]) / 2.0

        for i in range(n):
            nxt = 0 if i == n - 1 else i + 1
            if d[i] == 0.0:
                slopes[i] = 0.0
                slopes[nxt] = 0.0
            else:
                a = slopes[i] / d[i]
                b = slopes[nxt] / d[i]
                hyp = math.hypot(a, b)
                if hyp > 9.0:
                    t = 3.0 / hyp
                    slopes[i] = t * a * d[i]
                    slopes[nxt] = t * b * d[i]
        return slopes

    @property
    def num_points(self):
        return len(self.points)


class CyclicalWave:
    def __init__(self, shape: WaveShape):
        if shape.num_points < 2:
            raise ValueError("Shape must contain at least two points")
        self.shape = shape

    @property
    def name(self):
        return self.shape.name

    @property
    def num_points(self):
        return self.shape.num_points

    def _find_surrounding_points(self, phase):
        points = self.shape.points
        index = -1
        for i, p in enumerate(points):
            if p.time <= phase:
                index = i
        if index == -1:
            prev = points[-1]
            previous = WavePoint(prev.time - 1.0, prev.position, prev.slope)
            return previous, points[0]
        if index == len(points) - 1:
            nxt = points[0]
            following = WavePoint(nxt.time + 1.0, nxt.position, nxt.slope)
            return points[-1], following
        return points[index], points[index + 1]

    def get_position_and_velocity(self, simulation_time):
        phase = simulation_time % 1.0
        prev, nxt = self._find_surrounding_points(phase)
        if self.shape.interpolation_type == HERMITE:
            position, velocity = mu.hermite_interpolate_with_velocity(
                phase, prev.time, prev.position, prev.slope,
                nxt.time, nxt.position, nxt.slope,
            )
        else:
            position, velocity = mu.linear_interpolate_with_velocity(
                phase, prev.time, prev.position, nxt.time, nxt.position,
            )
        return mu.clamp(position, 0.0, 1.0), velocity

    def get_position(self, simulation_time):
        phase = simulation_time % 1.0
        prev, nxt = self._find_surrounding_points(phase)
        if self.shape.interpolation_type == HERMITE:
            position = mu.hermite_interpolate(
                phase, prev.time, prev.position, prev.slope,
                nxt.time, nxt.position, nxt.slope,
            )
        else:
            position = mu.linear_interpolate(
                phase, prev.time, prev.position, nxt.time, nxt.position,
            )
        return mu.clamp(position, 0.0, 1.0)


def create_repeated_wave(wave: CyclicalWave, repeats: int, new_name: str) -> CyclicalWave:
    if repeats <= 0:
        raise ValueError("Repeats must be positive")
    new_points = []
    time_scale = 1.0 / repeats
    for i in range(repeats):
        time_offset = i * time_scale
        for p in wave.shape.points:
            new_time = p.time * time_scale + time_offset
            if new_time < 1.0:
                new_slope = None if p.slope is None else p.slope * repeats
                new_points.append(WavePoint(new_time, p.position, new_slope))
    if len(new_points) < 2:
        first = new_points[0]
        new_points.append(WavePoint(TMAX, first.position, first.slope))
    new_points.sort(key=lambda p: p.time)
    return CyclicalWave(WaveShape(new_name, new_points, wave.shape.interpolation_type))


class JitterHandler:
    def __init__(self):
        self._previous_factor = 1.0
        self._current_factor = 1.0
        self._last_cycle = -1
        self.jitter = 0.0
        self.ease_in = 0.0

    def set_jitter(self, jitter):
        self.jitter = mu.clamp(jitter, 0.0, 1.0)

    def set_ease_in(self, ease_in):
        self.ease_in = mu.clamp(ease_in, 0.0, 1.0)

    def update(self, time):
        cycle = int(math.floor(time))
        if cycle != self._last_cycle:
            if self._last_cycle == -1:
                f = self._generate_factor()
                self._previous_factor = f
                self._current_factor = f
            else:
                self._previous_factor = self._current_factor
                self._current_factor = self._generate_factor()
            self._last_cycle = cycle

    def _generate_factor(self):
        if self.jitter == 0.0:
            return 1.0
        return 1.0 + mu.next_double(-self.jitter, self.jitter)

    def reset(self):
        self._last_cycle = -1
        self._previous_factor = 1.0
        self._current_factor = 1.0

    def get_interpolated_factor(self, time):
        phase = time % 1.0
        weight = 1.0 if self.ease_in == 0.0 else mu.clamp(phase / self.ease_in, 0.0, 1.0)
        raw = self._previous_factor + (self._current_factor - self._previous_factor) * weight
        return mu.clamp(raw, 1.0 - self.jitter, 1.0 + self.jitter)

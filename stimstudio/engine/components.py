"""Simulation components: ActivityManager, Timer, NiceSmoother, WaveManager.

Ported from Howl's ActivityComponents.kt and CyclicalWave.kt.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

from . import mathutils as mu
from .waves import CyclicalWave, JitterHandler


class Pulse(NamedTuple):
    amp_a: float = 0.0
    amp_b: float = 0.0
    freq_a: float = 0.0
    freq_b: float = 0.0


class ActivityComponent:
    def update(self, delta_time: float) -> None:  # pragma: no cover - interface
        pass

    def on_register(self) -> None:  # pragma: no cover - interface
        pass


class ActivityManager:
    def __init__(self):
        self._components: list[ActivityComponent] = []

    def register(self, component: ActivityComponent) -> None:
        self._components.append(component)
        component.on_register()

    def unregister(self, component: ActivityComponent) -> None:
        self._components.remove(component)

    def update(self, delta_time: float) -> None:
        for c in self._components:
            c.update(delta_time)


class Timer(ActivityComponent):
    INITIAL, RUNNING, FINISHED, PAUSED, CANCELLED = range(5)

    def __init__(self, duration_provider, repeating=False, on_trigger=lambda: None):
        if not callable(duration_provider):
            value = float(duration_provider)
            duration_provider = lambda: value
        self._duration_provider: Callable[[], float] = duration_provider
        self.repeating = repeating
        self._on_trigger = on_trigger
        self.duration = 0.0
        self.remaining_time = 0.0
        self.state = Timer.INITIAL

    @property
    def elapsed_time(self):
        return mu.clamp(self.duration - self.remaining_time, 0.0, self.duration)

    @property
    def progress(self):
        if self.duration <= 0.0:
            return 0.0
        return mu.clamp(self.elapsed_time / self.duration, 0.0, 1.0)

    @property
    def is_running(self):
        return self.state == Timer.RUNNING

    def _next_duration(self):
        value = self._duration_provider()
        assert value > 0.0, "Timer duration must be > 0.0"
        return value

    def update(self, delta_time):
        if self.state != Timer.RUNNING:
            return
        self.remaining_time -= delta_time
        while self.remaining_time <= 0.0:
            self._on_trigger()
            if not self.repeating:
                self.remaining_time = 0.0
                self.state = Timer.FINISHED
                return
            self.duration = self._next_duration()
            self.remaining_time += self.duration

    def start(self):
        self.reset()

    def reset(self):
        self.duration = self._next_duration()
        self.remaining_time = self.duration
        self.state = Timer.RUNNING

    def pause(self):
        self.state = Timer.PAUSED

    def resume(self):
        self.state = Timer.RUNNING

    def cancel(self):
        self.state = Timer.CANCELLED
        self.remaining_time = 0.0


class NiceSmoother(ActivityComponent):
    """Value that eases toward a target on an S-curve at a given rate."""

    def __init__(self, initial_value=0.0, value_range=(0.0, 1.0)):
        self.range = value_range
        self._value = mu.clamp(initial_value, value_range[0], value_range[1])
        self._rate = 1.0
        self._target = self._value
        self.velocity = 0.0
        self._start_value = initial_value
        self._start_velocity = 0.0
        self._duration = 0.0
        self._elapsed = 0.0
        self._is_transitioning = False
        self._on_reached = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = mu.clamp(v, self.range[0], self.range[1])

    @property
    def target(self):
        return self._target

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, value):
        self._rate = max(0.0001, value)
        if self._is_transitioning:
            self.set_target(self._target, on_reached=self._on_reached)

    def update(self, delta_time):
        if not self._is_transitioning:
            return
        self._elapsed += delta_time
        if self._elapsed >= self._duration:
            self._finish_transition()
            return
        pos, vel = mu.hermite_interpolate_with_velocity(
            self._elapsed, 0.0, self._start_value, self._start_velocity,
            self._duration, self._target, 0.0,
        )
        self.value = pos
        self.velocity = vel

    def set_target(self, target, rate=None, on_reached=None):
        self._on_reached = on_reached
        if rate is not None:
            self._rate = max(0.0001, rate)
        self._start_value = self._value
        self._start_velocity = self.velocity
        self._target = mu.clamp(target, self.range[0], self.range[1])
        self._elapsed = 0.0
        distance = abs(self._target - self._start_value)
        direction = math.copysign(1.0, self._target - self._start_value) if self._target != self._start_value else 0.0
        wrong_direction_penalty = (
            abs(self._start_velocity) / self._rate
            if self._start_velocity * direction < 0.0
            else 0.0
        )
        smooth_time = 0.2 / self._rate
        self._duration = (distance / self._rate) + wrong_direction_penalty + smooth_time
        self._is_transitioning = True

    def set_immediately(self, value):
        self.value = value
        self.velocity = 0.0
        self._target = mu.clamp(value, self.range[0], self.range[1])
        self._is_transitioning = False
        self._on_reached = None

    def _finish_transition(self):
        self.value = self._target
        self.velocity = 0.0
        self._is_transitioning = False
        callback = self._on_reached
        self._on_reached = None
        if callback is not None:
            callback()


class WaveManager(ActivityComponent):
    def __init__(self):
        self._waves: dict[str, CyclicalWave] = {}
        self._current_time = 0.0
        self._base_amplitude = 1.0
        self.base_speed = NiceSmoother(1.0, (0.01, 10.0))
        self._is_stopped = False
        self._stop_target_cycle = None
        self._stop_callback = None
        self._amp_jitter = JitterHandler()
        self._speed_jitter = JitterHandler()

    @property
    def current_amplitude(self):
        jitter = self._amp_jitter.jitter
        safe_base = self._base_amplitude if jitter == 0.0 else self._base_amplitude / (1.0 + jitter)
        return mu.clamp(safe_base * self._amp_jitter.get_interpolated_factor(self._current_time), 0.0, 1.0)

    @property
    def current_speed(self):
        return self.base_speed.value * self._speed_jitter.get_interpolated_factor(self._current_time)

    def add_wave(self, wave, name=None):
        self._waves[name or wave.name] = wave

    def get_wave(self, name):
        return self._waves[name]

    def update(self, delta_time):
        assert delta_time >= 0.0
        if self._is_stopped:
            return
        self.base_speed.update(delta_time)
        playback_time = delta_time * self.current_speed
        if self._stop_target_cycle is not None and self._current_time + playback_time >= self._stop_target_cycle:
            self._current_time = self._stop_target_cycle
            self._is_stopped = True
            cb = self._stop_callback
            if cb is not None:
                cb()
        else:
            self._current_time += playback_time
        self._amp_jitter.update(self._current_time)
        self._speed_jitter.update(self._current_time)

    def get_position_and_velocity(self, name, apply_amplitude=False, clamp_result=True, offset=0.0):
        wave = self._waves[name]
        position, velocity = wave.get_position_and_velocity(self._current_time + offset)
        scale_pos = self.current_amplitude if apply_amplitude else 1.0
        scale_vel = self.current_amplitude * self.current_speed if apply_amplitude else self.current_speed
        scaled_pos = scale_pos * position
        scaled_vel = scale_vel * velocity
        if clamp_result:
            scaled_pos = mu.clamp(scaled_pos, 0.0, 1.0)
        return scaled_pos, scaled_vel

    def get_position(self, name, apply_amplitude=False, clamp_result=True, offset=0.0):
        wave = self._waves[name]
        scale = self.current_amplitude if apply_amplitude else 1.0
        result = scale * wave.get_position(self._current_time + offset)
        return mu.clamp(result, 0.0, 1.0) if clamp_result else result

    def stop_at_end_of_cycle(self, callback):
        self._stop_target_cycle = math.floor(self._current_time + 1.0)
        self._stop_callback = callback

    def stop_after_iterations(self, iterations, callback):
        self._stop_target_cycle = math.floor(self._current_time + iterations)
        self._stop_callback = callback

    def restart(self):
        self._stop_target_cycle = None
        self._is_stopped = False
        self._current_time = 0.0
        self._amp_jitter.reset()
        self._speed_jitter.reset()

    def set_speed(self, new_speed):
        self.base_speed.set_immediately(new_speed)

    def set_amplitude(self, new_amplitude):
        self._base_amplitude = new_amplitude

    def set_amplitude_jitter(self, jitter):
        self._amp_jitter.set_jitter(mu.clamp(jitter, 0.0, 1.0))

    def set_speed_jitter(self, jitter):
        self._speed_jitter.set_jitter(mu.clamp(jitter, 0.0, 1.0))

    def set_amplitude_jitter_ease_in(self, ease_in):
        self._amp_jitter.set_ease_in(ease_in)

    def set_speed_jitter_ease_in(self, ease_in):
        self._speed_jitter.set_ease_in(ease_in)

    def set_target_speed(self, target, rate=None, on_reached=None):
        self.base_speed.set_target(target, rate=rate, on_reached=on_reached)

    def get_target_speed(self):
        return self.base_speed.target
